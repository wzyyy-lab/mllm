//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_FusedPDAttention);

// NOTE: This file currently provides a correctness-first reference implementation.
// On real devices, a float+expf softmax over long histories (e.g. 4k) can be slow enough
// to trigger DSP watchdog timeouts. To make "wiring smoke tests" safer, we cap the
// reference kernel's past length by default. For real performance/accuracy, replace this
// reference with an HVX/HMX optimized kernel and remove/raise the cap as needed.
#ifndef MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST
#define MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST 256
#endif

#ifndef MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ
#define MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ 128
#endif

// Use 1-pass online softmax (running max + rescale) to avoid a 2nd K pass.
// This is still a scalar reference kernel, but it reduces memory traffic significantly for long decode histories.
#ifndef MLLM_FUSED_PD_ATTENTION_USE_ONLINE_SOFTMAX
#define MLLM_FUSED_PD_ATTENTION_USE_ONLINE_SOFTMAX 1
#endif

// v1: decode-row streaming softmax reference (fixed-point, no expf).
//
// This is still NOT an optimized HVX kernel, but it avoids float+expf and avoids
// materializing a full [L] scores buffer for decode, which helps for long histories (e.g. 4k).
namespace {
constexpr int kScoreFracBits = 24;          // score in log2 domain, Q8.24-ish
constexpr int kBetaFracBits = 32;           // beta multiplier in Q0.32
constexpr int kExpLutFracBits = 30;         // exp2 LUT output in Q2.30 (range [1,2))
constexpr int kAlphaFracBits = 30;          // (v_scale/out_scale) in Q2.30-ish
constexpr int32_t kMinDeltaLog2Q = -(32 << kScoreFracBits); // clamp exp2(delta) for delta <= -32
constexpr int kMaxHeadDim = 256;            // scratch bound for reference kernels

// 2^(i/256) in Q30 for i=0..255.
static const uint32_t kExp2FracLutQ30[256] = {
  1073741824, 1076653033, 1079572136, 1082499153, 1085434106, 1088377016, 1091327906, 1094286796,
  1097253708, 1100228665, 1103211687, 1106202798, 1109202018, 1112209370, 1115224875, 1118248556,
  1121280436, 1124320536, 1127368878, 1130425485, 1133490379, 1136563583, 1139645120, 1142735011,
  1145833280, 1148939949, 1152055042, 1155178580, 1158310587, 1161451085, 1164600099, 1167757650,
  1170923762, 1174098458, 1177281762, 1180473697, 1183674286, 1186883552, 1190101520, 1193328213,
  1196563654, 1199807867, 1203060876, 1206322705, 1209593378, 1212872918, 1216161350, 1219458698,
  1222764986, 1226080238, 1229404479, 1232737732, 1236080024, 1239431376, 1242791816, 1246161366,
  1249540052, 1252927899, 1256324931, 1259731174, 1263146652, 1266571390, 1270005413, 1273448747,
  1276901417, 1280363448, 1283834865, 1287315695, 1290805962, 1294305692, 1297814910, 1301333643,
  1304861917, 1308399756, 1311947188, 1315504238, 1319070932, 1322647296, 1326233356, 1329829140,
  1333434672, 1337049980, 1340675091, 1344310030, 1347954824, 1351609500, 1355274085, 1358948606,
  1362633090, 1366327563, 1370032052, 1373746586, 1377471191, 1381205894, 1384950723, 1388705706,
  1392470869, 1396246240, 1400031848, 1403827719, 1407633882, 1411450365, 1415277195, 1419114401,
  1422962010, 1426820052, 1430688553, 1434567544, 1438457051, 1442357104, 1446267730, 1450188960,
  1454120821, 1458063343, 1462016553, 1465980482, 1469955159, 1473940611, 1477936870, 1481943963,
  1485961921, 1489990772, 1494030547, 1498081275, 1502142985, 1506215708, 1510299473, 1514394310,
  1518500250, 1522617322, 1526745556, 1530884983, 1535035634, 1539197537, 1543370725, 1547555228,
  1551751076, 1555958300, 1560176931, 1564406999, 1568648537, 1572901575, 1577166143, 1581442275,
  1585730000, 1590029350, 1594340357, 1598663052, 1602997467, 1607343634, 1611701585, 1616071351,
  1620452965, 1624846459, 1629251865, 1633669214, 1638098541, 1642539877, 1646993254, 1651458706,
  1655936265, 1660425963, 1664927835, 1669441912, 1673968228, 1678506817, 1683057710, 1687620943,
  1692196547, 1696784557, 1701385007, 1705997930, 1710623359, 1715261330, 1719911875, 1724575029,
  1729250827, 1733939301, 1738640488, 1743354420, 1748081133, 1752820662, 1757573041, 1762338305,
  1767116489, 1771907628, 1776711757, 1781528911, 1786359126, 1791202437, 1796058879, 1800928489,
  1805811301, 1810707353, 1815616678, 1820539314, 1825475297, 1830424663, 1835387448, 1840363688,
  1845353420, 1850356681, 1855373507, 1860403934, 1865448001, 1870505744, 1875577199, 1880662405,
  1885761398, 1890874216, 1896000896, 1901141476, 1906295993, 1911464486, 1916646992, 1921843549,
  1927054196, 1932278970, 1937517909, 1942771053, 1948038440, 1953320108, 1958616096, 1963926443,
  1969251188, 1974590370, 1979944027, 1985312200, 1990694927, 1996092249, 2001504204, 2006930832,
  2012372174, 2017828268, 2023299156, 2028784876, 2034285470, 2039800978, 2045331439, 2050876895,
  2056437387, 2062012954, 2067603638, 2073209480, 2078830522, 2084466803, 2090118366, 2095785251,
  2101467502, 2107165158, 2112878262, 2118606857, 2124350982, 2130110682, 2135885998, 2141676973,
};

static inline int64_t div_round_nearest_i64(int64_t num, int64_t den) {
  if (den == 0) return 0;
  if (num >= 0) return (num + den / 2) / den;
  return (num - den / 2) / den;
}

// Compute (x * m_q30) >> 30 without 128-bit intermediates.
// m_q30 is unsigned Q30 in [0, 1<<30]. Truncates toward zero for negative x.
//
// Dynamic range note: in this kernel, |x| stays well below 2^63 because:
// - sum_exp_q30 <= (past_len + seq_len) * (1<<30)
// - out_num_q30 <= sum_exp_q30 * max(|v_delta|)
static inline int64_t mul_q30_trunc_i64(int64_t x, uint32_t m_q30) {
  if (m_q30 == 0 || x == 0) return 0;
  const uint64_t ax = (x < 0) ? (uint64_t)(-(uint64_t)x) : (uint64_t)x;
  const uint64_t lo = (uint32_t)(ax & 0xFFFFFFFFu);
  const uint64_t hi = ax >> 32;
  // (hi<<32 + lo) * m >> 30 = (hi*m)<<(32-30) + (lo*m)>>30 = (hi*m)<<2 + (lo*m)>>30
  const uint64_t res = ((hi * (uint64_t)m_q30) << 2) + ((lo * (uint64_t)m_q30) >> 30);
  return (x < 0) ? -(int64_t)res : (int64_t)res;
}

// exp2(delta) for delta<=0 in log2 domain, where delta is Q(kScoreFracBits).
// Returns Q30 in [0, 1<<30].
static inline uint32_t exp2_neg_log2_q_to_q30(int32_t delta_q) {
  if (delta_q <= kMinDeltaLog2Q) return 0;
  // integer part (floor) and remainder in [0, 2^kScoreFracBits).
  const int32_t ip = delta_q >> kScoreFracBits; // <= 0
  const int32_t rem = delta_q - (ip << kScoreFracBits);
  const uint32_t idx = (uint32_t)rem >> (kScoreFracBits - 8); // 0..255
  uint32_t v = kExp2FracLutQ30[idx];
  if (ip == 0) return v;
  const int32_t sh = -ip;
  if (sh >= 31) return 0;
  return v >> sh;
}
} // namespace

// op execute function declarations
template<typename TensorType>
GraphStatus fusedPDAttentionImpl(TensorType& out_attn,
                                 const TensorType& q,
                                 const TensorType& k_curr,
                                 const TensorType& v_curr,
                                 const TensorType& past_k_prefill,
                                 const TensorType& past_v_prefill,
                                 const TensorType& past_k_decode,
                                 const TensorType& past_v_decode,
                                 const TensorType& attention_mask,
                                 const TensorType& fusion_ctrl,
                                 const TensorType& ref_max_seq_override,
                                 const TensorType& ref_max_past_override,
                                 const TensorType& q_scale,
                                 const TensorType& q_zp,
                                 const TensorType& k_scale,
                                 const TensorType& k_zp,
                                 const TensorType& v_scale,
                                 const TensorType& v_zp,
                                 const TensorType& out_scale,
                                 const TensorType& out_zp);

// forward declaration of sample cost function
static float fusedPDAttentionCostFunc(const Op* op);

DEF_PACKAGE_OP((fusedPDAttentionImpl<Tensor>), "FusedPDAttention")

template<typename TensorType>
GraphStatus fusedPDAttentionImpl(TensorType& out_attn,
                                 const TensorType& q,
                                 const TensorType& k_curr,
                                 const TensorType& v_curr,
                                 const TensorType& past_k_prefill,
                                 const TensorType& past_v_prefill,
                                 const TensorType& past_k_decode,
                                 const TensorType& past_v_decode,
                                 const TensorType& /*attention_mask*/,
                                 const TensorType& fusion_ctrl,
                                 const TensorType& ref_max_seq_override,
                                 const TensorType& ref_max_past_override,
                                 const TensorType& q_scale,
                                 const TensorType& q_zp,
                                 const TensorType& k_scale,
                                 const TensorType& k_zp,
                                 const TensorType& v_scale,
                                 const TensorType& v_zp,
                                 const TensorType& out_scale,
                                 const TensorType& out_zp) {
  // Kernel C v2 (streaming/tiled, mask-free reference):
  // - Computes PD attention without reading attention_mask.
  // - Uses fusion_ctrl to determine active rows and past lengths.
  // - Uses fixed-point dot + exp2 LUT (no float expf) and streaming softmax (no O(L) scores buffer).
  //
  // This is still scalar C++ (not HVX/HMX optimized), but it avoids the most dangerous watchdog traps
  // (float+expf + materializing long score vectors) and makes compute scale with (m, past_len).

  out_attn.set_dims(q);

  // Expect Q and output to be uint16 (per-tensor asymmetric).
  if (q.get_dtype() != DType::QUInt16 || out_attn.get_dtype() != DType::QUInt16) { return GraphStatus::Success; }
  // Expect K/V to be uint8 (per-tensor symmetric represented as uint8 with zp=128).
  if (k_curr.get_dtype() != DType::QUInt8 || v_curr.get_dtype() != DType::QUInt8) { return GraphStatus::Success; }

  auto [b_q, h_attn, s_q, d_q] = q.dims();  // [B, H_attn, S, D]
  auto [b_k, h_kv, d_k, s_k] = k_curr.dims();  // [B, H_kv, D, S]
  auto [b_v, h_v, s_v, d_v] = v_curr.dims();  // [B, H_kv, S, D]
  // K cache is token-major: [B, H_kv, past_len, D]
  auto [b_pk0, h_pk0, past_len0, d_pk0] = past_k_prefill.dims();
  auto [b_pv0, h_pv0, past_lenv0, d_pv0] = past_v_prefill.dims(); // [B, H_kv, past_len, D]
  auto [b_pk1, h_pk1, past_len1, d_pk1] = past_k_decode.dims();
  auto [b_pv1, h_pv1, past_lenv1, d_pv1] = past_v_decode.dims();

  if (b_q != b_k || b_q != b_v) { return GraphStatus::Success; }
  if (h_kv != h_v) { return GraphStatus::Success; }
  if (d_q != d_k || d_q != d_v) { return GraphStatus::Success; }
  if (s_q != s_k || s_q != s_v) { return GraphStatus::Success; }
  if (h_kv != h_pk0 || h_kv != h_pv0 || h_kv != h_pk1 || h_kv != h_pv1) { return GraphStatus::Success; }
  if (d_k != d_pk0 || d_k != d_pv0 || d_k != d_pk1 || d_k != d_pv1) { return GraphStatus::Success; }
  if (past_len0 != past_lenv0 || past_len1 != past_lenv1) { return GraphStatus::Success; }
  if (past_len0 != past_len1) { return GraphStatus::Success; }
  if (b_pk0 != b_q || b_pv0 != b_q || b_pk1 != b_q || b_pv1 != b_q) { return GraphStatus::Success; }

  const int32_t past_len = static_cast<int32_t>(past_len0);
  const int32_t seq_len = static_cast<int32_t>(s_q);
  const int32_t head_dim = static_cast<int32_t>(d_q);
  const int32_t ctx_len = past_len + seq_len;
  if (seq_len <= 0 || head_dim <= 0 || ctx_len <= 0) { return GraphStatus::Success; }
  if (head_dim > kMaxHeadDim) { return GraphStatus::Success; }

  // Scalars
  const float q_scale_f = static_cast<float>(q_scale(0, 0, 0, 0));
  const int32_t q_zp_i = static_cast<int32_t>(q_zp(0, 0, 0, 0));
  const float k_scale_f = static_cast<float>(k_scale(0, 0, 0, 0));
  const int32_t k_zp_i = static_cast<int32_t>(k_zp(0, 0, 0, 0));
  const float v_scale_f = static_cast<float>(v_scale(0, 0, 0, 0));
  const int32_t v_zp_i = static_cast<int32_t>(v_zp(0, 0, 0, 0));
  const float out_scale_f = static_cast<float>(out_scale(0, 0, 0, 0));
  const int32_t out_zp_i = static_cast<int32_t>(out_zp(0, 0, 0, 0));
  if (q_scale_f == 0.f || k_scale_f == 0.f || v_scale_f == 0.f || out_scale_f == 0.f) { return GraphStatus::Success; }

  const int32_t prefill_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 0));
  const int32_t decode_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 1));
  const int32_t prefill_n_update = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 2));
  // Runtime guards for the reference kernel (independent scalar inputs; do not overload fusion_ctrl):
  // - 0  : use compile-time default (MLLM_FUSED_PD_ATTENTION_REF_MAX_{SEQ,PAST})
  // - >0 : cap allowed {seq_len,past_len} to this value (if larger, the op returns early/clamps)
  // - <0 : disable the corresponding guard entirely
  const int32_t ref_max_seq_override_i = static_cast<int32_t>(ref_max_seq_override(0, 0, 0, 0));
  const int32_t ref_max_past_override_i = static_cast<int32_t>(ref_max_past_override(0, 0, 0, 0));
  int32_t prefill_n_past = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 4));
  int32_t decode_n_past = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 5));
  if (prefill_n_past < 0) prefill_n_past = 0;
  if (decode_n_past < 0) decode_n_past = 0;

  int32_t eff_max_past = MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST;
  if (ref_max_past_override_i > 0) eff_max_past = ref_max_past_override_i;
  if (ref_max_past_override_i < 0) eff_max_past = 0; // 0 => disable
  const int32_t clamp_past = (eff_max_past > 0) ? std::min(past_len, eff_max_past) : past_len;
  if (prefill_n_past > clamp_past) prefill_n_past = clamp_past;
  if (decode_n_past > clamp_past) decode_n_past = clamp_past;

  const int32_t decode_idx = seq_len - 1;
  const int32_t max_prefill_rows = decode_idx;
  int32_t prefill_rows = prefill_n_update;
  if (prefill_rows < 0) prefill_rows = 0;
  if (prefill_rows > max_prefill_rows) prefill_rows = max_prefill_rows;
  const int32_t prefill_base = decode_idx - prefill_rows;

  const int32_t groups = (h_kv > 0) ? (h_attn / h_kv) : 1;
  if (groups <= 0) { return GraphStatus::Success; }

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(q.raw_data_const());
  const uint8_t* k_curr_ptr = reinterpret_cast<const uint8_t*>(k_curr.raw_data_const());
  const uint8_t* v_curr_ptr = reinterpret_cast<const uint8_t*>(v_curr.raw_data_const());

  const uint8_t* pk0_ptr = reinterpret_cast<const uint8_t*>(past_k_prefill.raw_data_const());
  const uint8_t* pv0_ptr = reinterpret_cast<const uint8_t*>(past_v_prefill.raw_data_const());
  const uint8_t* pk1_ptr = reinterpret_cast<const uint8_t*>(past_k_decode.raw_data_const());
  const uint8_t* pv1_ptr = reinterpret_cast<const uint8_t*>(past_v_decode.raw_data_const());

  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out_attn.raw_data());
  {
    const size_t total = (size_t)b_q * (size_t)h_attn * (size_t)seq_len * (size_t)head_dim;
    if (out_zp_i == 0) {
      std::memset(out_ptr, 0, total * sizeof(uint16_t));
    } else {
      const uint16_t fillv = (uint16_t)std::clamp(out_zp_i, 0, 65535);
      for (size_t i = 0; i < total; ++i) out_ptr[i] = fillv;
    }
  }

  // Safety guard for the reference implementation:
  // avoid accidentally running very large shapes on DSP when doing smoke tests.
  int32_t eff_max_seq = MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ;
  if (ref_max_seq_override_i > 0) eff_max_seq = ref_max_seq_override_i;
  if (ref_max_seq_override_i < 0) eff_max_seq = 0;
  if (eff_max_seq > 0 && seq_len > eff_max_seq) {
    return GraphStatus::Success;
  }
  if (eff_max_past > 0 && past_len > eff_max_past) {
    return GraphStatus::Success;
  }

  // Helper lambdas for indexing.
  auto q_off = [&](int32_t b, int32_t h, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_attn + (size_t)h) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };
  auto out_off = q_off;

  // k cache: [B,H_kv,Len,D] (token-major)
  auto kcache_off = [&](int32_t b, int32_t hk, int32_t d, int32_t t, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)len + (size_t)t) * (size_t)head_dim + (size_t)d;
  };
  // v cache: [B,H_kv,Len,D] (token-major)
  auto vcache_off = [&](int32_t b, int32_t hk, int32_t t, int32_t d, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)len + (size_t)t) * (size_t)head_dim + (size_t)d;
  };
  // k_curr: [B,H_kv,D,S]
  auto kcurr_off = [&](int32_t b, int32_t hk, int32_t d, int32_t s) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)head_dim + (size_t)d) * (size_t)seq_len + (size_t)s;
  };
  // v_curr: [B,H_kv,S,D]
  auto vcurr_off = [&](int32_t b, int32_t hk, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };

  const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

  // NOTE: This reference kernel avoids reading attention_mask to reduce bandwidth.
  // It reconstructs the allowed positions from (n_past, row index) matching PDFusionRunner::prepare_io().

  // Decode-row streaming softmax (v1): avoids float+expf and does not allocate an O(L) scores/probs buffer.
  //
  // Note: this is still scalar C++ and will not match HVX performance, but it is much safer for long decode histories.
  const double log2e = 1.44269504088896340736; // 1/ln(2)
  const double beta = (double)q_scale_f * (double)k_scale_f * (double)inv_sqrt_d * log2e; // score in log2 domain
  int64_t beta_q32 = (int64_t)llround(beta * (double)(1ULL << kBetaFracBits));
  if (beta_q32 == 0) beta_q32 = (beta >= 0.0) ? 1 : -1;
  const int64_t alpha_q30 = (int64_t)llround(((double)v_scale_f / (double)out_scale_f) * (double)(1ULL << kAlphaFracBits));

  // Shared scratch (small, reused across heads and rows). Avoid heap allocs in the hot path.
  int32_t decode_qd_buf[kMaxHeadDim];
  int64_t decode_out_num_buf[kMaxHeadDim];
  int32_t* decode_qd = decode_qd_buf;
  int64_t* decode_out_num = decode_out_num_buf;

  auto run_streaming_row = [&](int32_t b, int32_t h, int32_t row, bool is_decode_row, int32_t n_past,
                               int32_t curr_begin, int32_t curr_end) {
    const int32_t hk = h / groups;
    if (hk < 0 || hk >= (int32_t)h_kv) { return; }

    // q deltas (centered).
    for (int32_t d = 0; d < head_dim; ++d) {
      decode_qd[d] = (int32_t)q_ptr[q_off(b, h, row, d)] - q_zp_i;
    }

    const int32_t Lpast = std::min(std::max(n_past, 0), past_len);
    int32_t sb = curr_begin;
    int32_t se = curr_end;
    if (sb < 0) sb = 0;
    if (se >= seq_len) se = seq_len - 1;

    // Pass 1: find max score (log2 domain, QkScoreFracBits).
    int32_t max_score_q = std::numeric_limits<int32_t>::min();
    auto score_q_from_dot = [&](int64_t dot_i64) -> int32_t {
      // score_q = dot * beta * 2^kScoreFracBits = dot * beta_q32 >> (kBetaFracBits - kScoreFracBits)
      const int64_t prod = dot_i64 * beta_q32;
      const int64_t shifted = prod >> (kBetaFracBits - kScoreFracBits);
      if (shifted > (int64_t)std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
      if (shifted < (int64_t)std::numeric_limits<int32_t>::min()) return std::numeric_limits<int32_t>::min();
      return (int32_t)shifted;
    };

#if MLLM_FUSED_PD_ATTENTION_USE_ONLINE_SOFTMAX
    // Online 1-pass streaming softmax: reads each K/V once, rescales when max increases.
    std::memset(decode_out_num, 0, (size_t)head_dim * sizeof(int64_t));
    int64_t sum_exp_q30 = 0;

    auto ingest = [&](int32_t score_q, auto&& v_getter) {
      if (max_score_q == std::numeric_limits<int32_t>::min()) {
        max_score_q = score_q;
        sum_exp_q30 = (1LL << kExpLutFracBits);
        for (int32_t d = 0; d < head_dim; ++d) {
          const int32_t v_u = (int32_t)v_getter(d);
          decode_out_num[d] = (int64_t)(1LL << kExpLutFracBits) * (int64_t)(v_u - v_zp_i);
        }
        return;
      }

      if (score_q > max_score_q) {
        int32_t delta_q = max_score_q - score_q; // <= 0
        if (delta_q < kMinDeltaLog2Q) delta_q = kMinDeltaLog2Q;
        const uint32_t scale_q30 = exp2_neg_log2_q_to_q30(delta_q);
        sum_exp_q30 = mul_q30_trunc_i64(sum_exp_q30, scale_q30);
        for (int32_t d = 0; d < head_dim; ++d) decode_out_num[d] = mul_q30_trunc_i64(decode_out_num[d], scale_q30);
        max_score_q = score_q;

        // Add current element with weight=1.
        sum_exp_q30 += (1LL << kExpLutFracBits);
        for (int32_t d = 0; d < head_dim; ++d) {
          const int32_t v_u = (int32_t)v_getter(d);
          decode_out_num[d] += (int64_t)(1LL << kExpLutFracBits) * (int64_t)(v_u - v_zp_i);
        }
        return;
      }

      int32_t delta_q = score_q - max_score_q; // <= 0
      if (delta_q < kMinDeltaLog2Q) return;
      const uint32_t w_q30 = exp2_neg_log2_q_to_q30(delta_q);
      if (w_q30 == 0) return;
      sum_exp_q30 += (int64_t)w_q30;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t v_u = (int32_t)v_getter(d);
        decode_out_num[d] += (int64_t)w_q30 * (int64_t)(v_u - v_zp_i);
      }
    };

    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)(is_decode_row ? pk1_ptr[kcache_off(b, hk, d, t, past_len)]
                                                   : pk0_ptr[kcache_off(b, hk, d, t, past_len)]);
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      ingest(score_q, [&](int32_t d) -> uint8_t {
        return is_decode_row ? pv1_ptr[vcache_off(b, hk, t, d, past_len)] : pv0_ptr[vcache_off(b, hk, t, d, past_len)];
      });
    }
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)];
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      ingest(score_q, [&](int32_t d) -> uint8_t { return v_curr_ptr[vcurr_off(b, hk, s, d)]; });
    }

    if (sum_exp_q30 <= 0) return;

    for (int32_t d = 0; d < head_dim; ++d) {
      const int64_t avg_v_delta = div_round_nearest_i64(decode_out_num[d], sum_exp_q30); // roughly [-128,127]
      const int64_t scaled = avg_v_delta * alpha_q30;
      int64_t out_delta = (scaled >= 0) ? ((scaled + (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits)
                                        : ((scaled - (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits);
      int64_t out_u = (int64_t)out_zp_i + out_delta;
      if (out_u < 0) out_u = 0;
      if (out_u > 65535) out_u = 65535;
      out_ptr[out_off(b, h, row, d)] = (uint16_t)out_u;
    }
    return;
#endif

    // past positions
    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)(is_decode_row ? pk1_ptr[kcache_off(b, hk, d, t, past_len)]
                                                   : pk0_ptr[kcache_off(b, hk, d, t, past_len)]);
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      if (score_q > max_score_q) max_score_q = score_q;
    }
    // current positions (causal segment)
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)];
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      if (score_q > max_score_q) max_score_q = score_q;
    }

    if (max_score_q == std::numeric_limits<int32_t>::min()) {
      return;
    }

    // Pass 2: sum exp and accumulate weighted V numerators (still in int domain).
    std::memset(decode_out_num, 0, (size_t)head_dim * sizeof(int64_t));
    int64_t sum_exp = 0;
    auto accumulate = [&](int32_t score_q, auto&& v_getter) {
      int64_t delta_q64 = (int64_t)score_q - (int64_t)max_score_q; // <= 0 (but guard against int32 overflow)
      if (delta_q64 < (int64_t)kMinDeltaLog2Q) delta_q64 = (int64_t)kMinDeltaLog2Q;
      if (delta_q64 > 0) delta_q64 = 0;
      const uint32_t e_q30 = exp2_neg_log2_q_to_q30((int32_t)delta_q64);
      sum_exp += (int64_t)e_q30;
      if (e_q30 == 0) return;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t v_u = (int32_t)v_getter(d);
        decode_out_num[d] += (int64_t)e_q30 * (int64_t)(v_u - v_zp_i);
      }
    };

    // past V
    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)(is_decode_row ? pk1_ptr[kcache_off(b, hk, d, t, past_len)]
                                                   : pk0_ptr[kcache_off(b, hk, d, t, past_len)]);
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      accumulate(score_q, [&](int32_t d) -> uint8_t {
        return is_decode_row ? pv1_ptr[vcache_off(b, hk, t, d, past_len)] : pv0_ptr[vcache_off(b, hk, t, d, past_len)];
      });
    }

    // current V
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)];
        dot += (int64_t)decode_qd[d] * (int64_t)(k_u - k_zp_i);
      }
      const int32_t score_q = score_q_from_dot(dot);
      accumulate(score_q, [&](int32_t d) -> uint8_t { return v_curr_ptr[vcurr_off(b, hk, s, d)]; });
    }

    if (sum_exp <= 0) {
      return;
    }

    // Normalize by sum_exp and requantize to uint16 output.
    for (int32_t d = 0; d < head_dim; ++d) {
      const int64_t avg_v_delta = div_round_nearest_i64(decode_out_num[d], sum_exp); // roughly [-128,127]
      const int64_t scaled = avg_v_delta * alpha_q30;
      int64_t out_delta = (scaled >= 0) ? ((scaled + (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits)
                                        : ((scaled - (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits);
      int64_t out_u = (int64_t)out_zp_i + out_delta;
      if (out_u < 0) out_u = 0;
      if (out_u > 65535) out_u = 65535;
      out_ptr[out_off(b, h, row, d)] = (uint16_t)out_u;
    }
  };

  // Execute requested rows.
  if (prefill_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) {
        for (int32_t t = 0; t < prefill_rows; ++t) {
          const int32_t row = prefill_base + t;
          // Right-aligned causal segment within current chunk.
          run_streaming_row(b, h, row, /*is_decode_row=*/false, prefill_n_past, prefill_base, row);
        }
      }
    }
  }
  if (decode_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) {
        run_streaming_row(b, h, decode_idx, /*is_decode_row=*/true, decode_n_past, decode_idx, decode_idx);
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float fusedPDAttentionCostFunc(const Op* /*op*/) { return 0.0f; }

END_PKG_OP_DEFINITION(PKG_FusedPDAttention);

// -----------------------------------------------------------------------------
// Kernel C v3: k=4 decode rows (multi-request) + prefill rows in one op.
//
// This variant is intended for future PD-k graphs (e.g. k=4). It keeps the same core
// streaming softmax (exp2 LUT) but computes attention for:
// - Prefill rows: [prefill_base .. decode_base-1] (m rows, right-aligned)
// - Decode rows:  [decode_base .. decode_base+3] (4 rows, one per request)
//
// Each decode row has its own KV cache input and past length (decode_past_lens[4]).
// No attention_mask is read.
BEGIN_PKG_OP_DEFINITION(PKG_FusedPDAttentionK4);

template<typename TensorType>
GraphStatus fusedPDAttentionK4Impl(TensorType& out_attn,
                                   const TensorType& q,
                                   const TensorType& k_curr,
                                   const TensorType& v_curr,
                                   const TensorType& past_k_prefill,
                                   const TensorType& past_v_prefill,
                                   const TensorType& past_k_dec0,
                                   const TensorType& past_v_dec0,
                                   const TensorType& past_k_dec1,
                                   const TensorType& past_v_dec1,
                                   const TensorType& past_k_dec2,
                                   const TensorType& past_v_dec2,
                                   const TensorType& past_k_dec3,
                                   const TensorType& past_v_dec3,
                                   const TensorType& /*attention_mask*/,
                                   const TensorType& fusion_ctrl,
                                   const TensorType& decode_past_lens,
                                   const TensorType& ref_max_seq_override,
                                   const TensorType& ref_max_past_override,
                                   const TensorType& q_scale,
                                   const TensorType& q_zp,
                                   const TensorType& k_scale,
                                   const TensorType& k_zp,
                                   const TensorType& v_scale,
                                   const TensorType& v_zp,
                                   const TensorType& out_scale,
                                   const TensorType& out_zp);

static float fusedPDAttentionK4CostFunc(const Op* op);

DEF_PACKAGE_OP((fusedPDAttentionK4Impl<Tensor>), "FusedPDAttentionK4")

template<typename TensorType>
GraphStatus fusedPDAttentionK4Impl(TensorType& out_attn,
                                   const TensorType& q,
                                   const TensorType& k_curr,
                                   const TensorType& v_curr,
                                   const TensorType& past_k_prefill,
                                   const TensorType& past_v_prefill,
                                   const TensorType& past_k_dec0,
                                   const TensorType& past_v_dec0,
                                   const TensorType& past_k_dec1,
                                   const TensorType& past_v_dec1,
                                   const TensorType& past_k_dec2,
                                   const TensorType& past_v_dec2,
                                   const TensorType& past_k_dec3,
                                   const TensorType& past_v_dec3,
                                   const TensorType& /*attention_mask*/,
                                   const TensorType& fusion_ctrl,
                                   const TensorType& decode_past_lens,
                                   const TensorType& ref_max_seq_override,
                                   const TensorType& ref_max_past_override,
                                   const TensorType& q_scale,
                                   const TensorType& q_zp,
                                   const TensorType& k_scale,
                                   const TensorType& k_zp,
                                   const TensorType& v_scale,
                                   const TensorType& v_zp,
                                   const TensorType& out_scale,
                                   const TensorType& out_zp) {
  out_attn.set_dims(q);

  if (q.get_dtype() != DType::QUInt16 || out_attn.get_dtype() != DType::QUInt16) { return GraphStatus::Success; }
  if (k_curr.get_dtype() != DType::QUInt8 || v_curr.get_dtype() != DType::QUInt8) { return GraphStatus::Success; }

  auto [b_q, h_attn, s_q, d_q] = q.dims();               // [B, H_attn, S, D]
  auto [b_k, h_kv, d_k, s_k] = k_curr.dims();            // [B, H_kv, D, S]
  auto [b_v, h_v, s_v, d_v] = v_curr.dims();             // [B, H_kv, S, D]
  auto [b_pk0, h_pk0, past_len0, d_pk0] = past_k_prefill.dims();
  auto [b_pv0, h_pv0, past_lenv0, d_pv0] = past_v_prefill.dims();
  auto [b_pk1, h_pk1, past_len1, d_pk1] = past_k_dec0.dims();
  auto [b_pv1, h_pv1, past_lenv1, d_pv1] = past_v_dec0.dims();
  auto [b_pk2, h_pk2, past_len2, d_pk2] = past_k_dec1.dims();
  auto [b_pv2, h_pv2, past_lenv2, d_pv2] = past_v_dec1.dims();
  auto [b_pk3, h_pk3, past_len3, d_pk3] = past_k_dec2.dims();
  auto [b_pv3, h_pv3, past_lenv3, d_pv3] = past_v_dec2.dims();
  auto [b_pk4, h_pk4, past_len4, d_pk4] = past_k_dec3.dims();
  auto [b_pv4, h_pv4, past_lenv4, d_pv4] = past_v_dec3.dims();

  if (b_q != b_k || b_q != b_v) { return GraphStatus::Success; }
  if (h_kv != h_v) { return GraphStatus::Success; }
  if (d_q != d_k || d_q != d_v) { return GraphStatus::Success; }
  if (s_q != s_k || s_q != s_v) { return GraphStatus::Success; }
  if (h_kv != h_pk0 || h_kv != h_pv0) { return GraphStatus::Success; }
  if (d_k != d_pk0 || d_k != d_pv0) { return GraphStatus::Success; }
  if (past_len0 != past_lenv0) { return GraphStatus::Success; }
  if (past_len0 != past_len1 || past_len0 != past_len2 || past_len0 != past_len3 || past_len0 != past_len4) { return GraphStatus::Success; }
  if (past_len1 != past_lenv1 || past_len2 != past_lenv2 || past_len3 != past_lenv3 || past_len4 != past_lenv4) { return GraphStatus::Success; }
  if (h_kv != h_pk1 || h_kv != h_pv1 || h_kv != h_pk2 || h_kv != h_pv2 || h_kv != h_pk3 || h_kv != h_pv3 || h_kv != h_pk4 ||
      h_kv != h_pv4) {
    return GraphStatus::Success;
  }
  if (d_k != d_pk1 || d_k != d_pv1 || d_k != d_pk2 || d_k != d_pv2 || d_k != d_pk3 || d_k != d_pv3 || d_k != d_pk4 || d_k != d_pv4) {
    return GraphStatus::Success;
  }

  const int32_t seq_len = (int32_t)s_q;
  const int32_t head_dim = (int32_t)d_q;
  const int32_t past_len = (int32_t)past_len0;
  if (seq_len < 4) { return GraphStatus::Success; }
  if (head_dim > kMaxHeadDim) { return GraphStatus::Success; }

  const float q_scale_f = static_cast<float>(q_scale(0, 0, 0, 0));
  const int32_t q_zp_i = static_cast<int32_t>(q_zp(0, 0, 0, 0));
  const float k_scale_f = static_cast<float>(k_scale(0, 0, 0, 0));
  const int32_t k_zp_i = static_cast<int32_t>(k_zp(0, 0, 0, 0));
  const float v_scale_f = static_cast<float>(v_scale(0, 0, 0, 0));
  const int32_t v_zp_i = static_cast<int32_t>(v_zp(0, 0, 0, 0));
  const float out_scale_f = static_cast<float>(out_scale(0, 0, 0, 0));
  const int32_t out_zp_i = static_cast<int32_t>(out_zp(0, 0, 0, 0));
  if (q_scale_f == 0.f || k_scale_f == 0.f || v_scale_f == 0.f || out_scale_f == 0.f) { return GraphStatus::Success; }

  const int32_t prefill_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 0));
  const int32_t decode_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 1));
  const int32_t prefill_n_update = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 2));
  int32_t prefill_n_past = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 4));
  if (prefill_n_past < 0) prefill_n_past = 0;
  if (prefill_n_past > past_len) prefill_n_past = past_len;

  const int32_t ref_max_seq_override_i = static_cast<int32_t>(ref_max_seq_override(0, 0, 0, 0));
  const int32_t ref_max_past_override_i = static_cast<int32_t>(ref_max_past_override(0, 0, 0, 0));

  int32_t eff_max_past = MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST;
  if (ref_max_past_override_i > 0) eff_max_past = ref_max_past_override_i;
  if (ref_max_past_override_i < 0) eff_max_past = 0; // 0 => disable
  const int32_t clamp_past = (eff_max_past > 0) ? std::min(past_len, eff_max_past) : past_len;
  if (prefill_n_past > clamp_past) prefill_n_past = clamp_past;

  // decode_past_lens: int32[4]
  int32_t dec_past[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    dec_past[i] = static_cast<int32_t>(decode_past_lens(0, 0, 0, i));
    if (dec_past[i] < 0) dec_past[i] = 0;
    if (dec_past[i] > past_len) dec_past[i] = past_len;
    if (dec_past[i] > clamp_past) dec_past[i] = clamp_past;
  }

  const int32_t decode_base = seq_len - 4;
  int32_t prefill_rows = prefill_n_update;
  if (prefill_rows < 0) prefill_rows = 0;
  if (prefill_rows > decode_base) prefill_rows = decode_base;
  const int32_t prefill_base = decode_base - prefill_rows;

  const int32_t groups = (h_kv > 0) ? ((int32_t)h_attn / (int32_t)h_kv) : 1;
  if (groups <= 0) { return GraphStatus::Success; }

  int32_t eff_max_seq = MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ;
  if (ref_max_seq_override_i > 0) eff_max_seq = ref_max_seq_override_i;
  if (ref_max_seq_override_i < 0) eff_max_seq = 0;
  if (eff_max_seq > 0 && seq_len > eff_max_seq) {
    return GraphStatus::Success;
  }
  if (eff_max_past > 0 && past_len > eff_max_past) {
    return GraphStatus::Success;
  }

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(q.raw_data_const());
  const uint8_t* k_curr_ptr = reinterpret_cast<const uint8_t*>(k_curr.raw_data_const());
  const uint8_t* v_curr_ptr = reinterpret_cast<const uint8_t*>(v_curr.raw_data_const());

  const uint8_t* pk0_ptr = reinterpret_cast<const uint8_t*>(past_k_prefill.raw_data_const());
  const uint8_t* pv0_ptr = reinterpret_cast<const uint8_t*>(past_v_prefill.raw_data_const());

  const uint8_t* pk_dec[4] = {
      reinterpret_cast<const uint8_t*>(past_k_dec0.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_k_dec1.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_k_dec2.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_k_dec3.raw_data_const()),
  };
  const uint8_t* pv_dec[4] = {
      reinterpret_cast<const uint8_t*>(past_v_dec0.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_v_dec1.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_v_dec2.raw_data_const()),
      reinterpret_cast<const uint8_t*>(past_v_dec3.raw_data_const()),
  };

  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out_attn.raw_data());
  const size_t total = (size_t)b_q * (size_t)h_attn * (size_t)seq_len * (size_t)head_dim;
  if (out_zp_i == 0) {
    std::memset(out_ptr, 0, total * sizeof(uint16_t));
  } else {
    const uint16_t fillv = (uint16_t)std::clamp(out_zp_i, 0, 65535);
    for (size_t i = 0; i < total; ++i) out_ptr[i] = fillv;
  }

  auto q_off = [&](int32_t b, int32_t h, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_attn + (size_t)h) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };
  auto out_off = q_off;
  auto kcache_off = [&](int32_t b, int32_t hk, int32_t d, int32_t t, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)len + (size_t)t) * (size_t)head_dim + (size_t)d;
  };
  auto vcache_off = [&](int32_t b, int32_t hk, int32_t t, int32_t d, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)len + (size_t)t) * (size_t)head_dim + (size_t)d;
  };
  auto kcurr_off = [&](int32_t b, int32_t hk, int32_t d, int32_t s) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)head_dim + (size_t)d) * (size_t)seq_len + (size_t)s;
  };
  auto vcurr_off = [&](int32_t b, int32_t hk, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };

  const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
  const double log2e = 1.44269504088896340736;
  const double beta = (double)q_scale_f * (double)k_scale_f * (double)inv_sqrt_d * log2e;
  int64_t beta_q32 = (int64_t)llround(beta * (double)(1ULL << kBetaFracBits));
  if (beta_q32 == 0) beta_q32 = (beta >= 0.0) ? 1 : -1;
  const int64_t alpha_q30 = (int64_t)llround(((double)v_scale_f / (double)out_scale_f) * (double)(1ULL << kAlphaFracBits));

  // Shared scratch (small, reused across heads and rows). Avoid heap allocs in the hot path.
  int32_t qd_buf[kMaxHeadDim];
  int64_t out_num_buf[kMaxHeadDim];
  int32_t* qd = qd_buf;
  int64_t* out_num = out_num_buf;

  auto score_q_from_dot = [&](int64_t dot_i64) -> int32_t {
    const int64_t prod = dot_i64 * beta_q32;
    const int64_t shifted = prod >> (kBetaFracBits - kScoreFracBits);
    if (shifted > (int64_t)std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
    if (shifted < (int64_t)std::numeric_limits<int32_t>::min()) return std::numeric_limits<int32_t>::min();
    return (int32_t)shifted;
  };

  auto run_row = [&](int32_t b, int32_t h, int32_t row, const uint8_t* pk, const uint8_t* pv, int32_t n_past, int32_t curr_begin,
                     int32_t curr_end) {
    const int32_t hk = h / groups;
    if (hk < 0 || hk >= (int32_t)h_kv) return;
    int32_t Lpast = std::min(std::max(n_past, 0), past_len);
    int32_t sb = curr_begin;
    int32_t se = curr_end;
    if (sb < 0) sb = 0;
    if (se >= seq_len) se = seq_len - 1;
    if (se < sb) { sb = 1; se = 0; } // empty

    for (int32_t d = 0; d < head_dim; ++d) qd[d] = (int32_t)q_ptr[q_off(b, h, row, d)] - q_zp_i;

#if MLLM_FUSED_PD_ATTENTION_USE_ONLINE_SOFTMAX
    // Online 1-pass streaming softmax: reads each K/V once, rescales when max increases.
    int32_t max_score_q = std::numeric_limits<int32_t>::min();
    int64_t sum_exp_q30 = 0;
    std::memset(out_num, 0, (size_t)head_dim * sizeof(int64_t));

    auto ingest = [&](int32_t score_q, auto&& v_getter) {
      if (max_score_q == std::numeric_limits<int32_t>::min()) {
        max_score_q = score_q;
        sum_exp_q30 = (1LL << kExpLutFracBits);
        for (int32_t d = 0; d < head_dim; ++d) out_num[d] = (int64_t)(1LL << kExpLutFracBits) * (int64_t)((int32_t)v_getter(d) - v_zp_i);
        return;
      }
      if (score_q > max_score_q) {
        int32_t delta_q = max_score_q - score_q;
        if (delta_q < kMinDeltaLog2Q) delta_q = kMinDeltaLog2Q;
        const uint32_t scale_q30 = exp2_neg_log2_q_to_q30(delta_q);
        sum_exp_q30 = mul_q30_trunc_i64(sum_exp_q30, scale_q30);
        for (int32_t d = 0; d < head_dim; ++d) out_num[d] = mul_q30_trunc_i64(out_num[d], scale_q30);
        max_score_q = score_q;
        sum_exp_q30 += (1LL << kExpLutFracBits);
        for (int32_t d = 0; d < head_dim; ++d) out_num[d] += (int64_t)(1LL << kExpLutFracBits) * (int64_t)((int32_t)v_getter(d) - v_zp_i);
        return;
      }
      int32_t delta_q = score_q - max_score_q;
      if (delta_q < kMinDeltaLog2Q) return;
      const uint32_t w_q30 = exp2_neg_log2_q_to_q30(delta_q);
      if (w_q30 == 0) return;
      sum_exp_q30 += (int64_t)w_q30;
      for (int32_t d = 0; d < head_dim; ++d) out_num[d] += (int64_t)w_q30 * (int64_t)((int32_t)v_getter(d) - v_zp_i);
    };

    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)pk[kcache_off(b, hk, d, t, past_len)] - k_zp_i);
      ingest(score_q_from_dot(dot), [&](int32_t d) -> uint8_t { return pv[vcache_off(b, hk, t, d, past_len)]; });
    }
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)] - k_zp_i);
      ingest(score_q_from_dot(dot), [&](int32_t d) -> uint8_t { return v_curr_ptr[vcurr_off(b, hk, s, d)]; });
    }
    if (sum_exp_q30 <= 0) return;

    for (int32_t d = 0; d < head_dim; ++d) {
      const int64_t avg_v_delta = div_round_nearest_i64(out_num[d], sum_exp_q30);
      const int64_t scaled = avg_v_delta * alpha_q30;
      int64_t out_delta = (scaled >= 0) ? ((scaled + (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits)
                                        : ((scaled - (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits);
      int64_t out_u = (int64_t)out_zp_i + out_delta;
      if (out_u < 0) out_u = 0;
      if (out_u > 65535) out_u = 65535;
      out_ptr[out_off(b, h, row, d)] = (uint16_t)out_u;
    }
    return;
#endif

    int32_t max_score_q = std::numeric_limits<int32_t>::min();
    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)pk[kcache_off(b, hk, d, t, past_len)] - k_zp_i);
      max_score_q = std::max(max_score_q, score_q_from_dot(dot));
    }
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)] - k_zp_i);
      max_score_q = std::max(max_score_q, score_q_from_dot(dot));
    }
    if (max_score_q == std::numeric_limits<int32_t>::min()) return;

    std::memset(out_num, 0, (size_t)head_dim * sizeof(int64_t));
    int64_t sum_exp = 0;
    auto accum = [&](int32_t score_q, auto&& v_getter) {
      int64_t delta_q64 = (int64_t)score_q - (int64_t)max_score_q;
      if (delta_q64 < (int64_t)kMinDeltaLog2Q) delta_q64 = (int64_t)kMinDeltaLog2Q;
      if (delta_q64 > 0) delta_q64 = 0;
      const uint32_t e_q30 = exp2_neg_log2_q_to_q30((int32_t)delta_q64);
      sum_exp += (int64_t)e_q30;
      if (e_q30 == 0) return;
      for (int32_t d = 0; d < head_dim; ++d) out_num[d] += (int64_t)e_q30 * (int64_t)((int32_t)v_getter(d) - v_zp_i);
    };
    for (int32_t t = 0; t < Lpast; ++t) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)pk[kcache_off(b, hk, d, t, past_len)] - k_zp_i);
      const int32_t score_q = score_q_from_dot(dot);
      accum(score_q, [&](int32_t d) -> uint8_t { return pv[vcache_off(b, hk, t, d, past_len)]; });
    }
    for (int32_t s = sb; s <= se; ++s) {
      int64_t dot = 0;
      for (int32_t d = 0; d < head_dim; ++d) dot += (int64_t)qd[d] * (int64_t)((int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)] - k_zp_i);
      const int32_t score_q = score_q_from_dot(dot);
      accum(score_q, [&](int32_t d) -> uint8_t { return v_curr_ptr[vcurr_off(b, hk, s, d)]; });
    }
    if (sum_exp <= 0) return;

    for (int32_t d = 0; d < head_dim; ++d) {
      const int64_t avg_v_delta = div_round_nearest_i64(out_num[d], sum_exp);
      const int64_t scaled = avg_v_delta * alpha_q30;
      int64_t out_delta = (scaled >= 0) ? ((scaled + (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits)
                                        : ((scaled - (1LL << (kAlphaFracBits - 1))) >> kAlphaFracBits);
      int64_t out_u = (int64_t)out_zp_i + out_delta;
      if (out_u < 0) out_u = 0;
      if (out_u > 65535) out_u = 65535;
      out_ptr[out_off(b, h, row, d)] = (uint16_t)out_u;
    }
  };

  if (prefill_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) {
        for (int32_t t = 0; t < prefill_rows; ++t) {
          const int32_t row = prefill_base + t;
          run_row(b, h, row, pk0_ptr, pv0_ptr, prefill_n_past, prefill_base, row);
        }
      }
    }
  }

  if (decode_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) {
        for (int i = 0; i < 4; ++i) {
          const int32_t row = decode_base + i;
          // Each decode row attends only to its own past + itself (no decode<->decode visibility).
          run_row(b, h, row, pk_dec[i], pv_dec[i], dec_past[i], row, row);
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float fusedPDAttentionK4CostFunc(const Op* /*op*/) { return 0.0f; }

END_PKG_OP_DEFINITION(PKG_FusedPDAttentionK4);

// -----------------------------------------------------------------------------
// System-level mask removal variants:
//
// These ops are identical to the mask-taking variants, except their signatures
// do not include attention_mask. This allows PD graphs to remove the large
// attention_mask input edge entirely (host no longer binds/syncs/writes it).
//
// NOTE: The underlying reference implementations do not read attention_mask
// anyway, so we forward to the existing implementations with a dummy tensor.

BEGIN_PKG_OP_DEFINITION(PKG_FusedPDAttentionNoMask);

template<typename TensorType>
GraphStatus fusedPDAttentionNoMaskImpl(TensorType& out_attn,
                                       const TensorType& q,
                                       const TensorType& k_curr,
                                       const TensorType& v_curr,
                                       const TensorType& past_k_prefill,
                                       const TensorType& past_v_prefill,
                                       const TensorType& past_k_decode,
                                       const TensorType& past_v_decode,
                                       const TensorType& fusion_ctrl,
                                       const TensorType& ref_max_seq_override,
                                       const TensorType& ref_max_past_override,
                                       const TensorType& q_scale,
                                       const TensorType& q_zp,
                                       const TensorType& k_scale,
                                       const TensorType& k_zp,
                                       const TensorType& v_scale,
                                       const TensorType& v_zp,
                                       const TensorType& out_scale,
                                       const TensorType& out_zp) {
  // Forward to the mask-taking implementation; attention_mask is ignored there.
  return fusedPDAttentionImpl(out_attn, q, k_curr, v_curr, past_k_prefill, past_v_prefill, past_k_decode, past_v_decode,
                             /*attention_mask=*/q, fusion_ctrl, ref_max_seq_override, ref_max_past_override, q_scale, q_zp, k_scale, k_zp,
                             v_scale, v_zp, out_scale, out_zp);
}

DEF_PACKAGE_OP((fusedPDAttentionNoMaskImpl<Tensor>), "FusedPDAttentionNoMask")

END_PKG_OP_DEFINITION(PKG_FusedPDAttentionNoMask);

BEGIN_PKG_OP_DEFINITION(PKG_FusedPDAttentionK4NoMask);

template<typename TensorType>
GraphStatus fusedPDAttentionK4NoMaskImpl(TensorType& out_attn,
                                         const TensorType& q,
                                         const TensorType& k_curr,
                                         const TensorType& v_curr,
                                         const TensorType& past_k_prefill,
                                         const TensorType& past_v_prefill,
                                         const TensorType& past_k_dec0,
                                         const TensorType& past_v_dec0,
                                         const TensorType& past_k_dec1,
                                         const TensorType& past_v_dec1,
                                         const TensorType& past_k_dec2,
                                         const TensorType& past_v_dec2,
                                         const TensorType& past_k_dec3,
                                         const TensorType& past_v_dec3,
                                         const TensorType& fusion_ctrl,
                                         const TensorType& decode_past_lens,
                                         const TensorType& ref_max_seq_override,
                                         const TensorType& ref_max_past_override,
                                         const TensorType& q_scale,
                                         const TensorType& q_zp,
                                         const TensorType& k_scale,
                                         const TensorType& k_zp,
                                         const TensorType& v_scale,
                                         const TensorType& v_zp,
                                         const TensorType& out_scale,
                                         const TensorType& out_zp) {
  // Forward to the mask-taking implementation; attention_mask is ignored there.
  return fusedPDAttentionK4Impl(out_attn, q, k_curr, v_curr, past_k_prefill, past_v_prefill, past_k_dec0, past_v_dec0, past_k_dec1,
                               past_v_dec1, past_k_dec2, past_v_dec2, past_k_dec3, past_v_dec3,
                               /*attention_mask=*/q, fusion_ctrl, decode_past_lens, ref_max_seq_override, ref_max_past_override, q_scale, q_zp,
                               k_scale, k_zp, v_scale, v_zp, out_scale, out_zp);
}

DEF_PACKAGE_OP((fusedPDAttentionK4NoMaskImpl<Tensor>), "FusedPDAttentionK4NoMask")

END_PKG_OP_DEFINITION(PKG_FusedPDAttentionK4NoMask);
