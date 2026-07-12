#include <cute/tensor.hpp>

#include <cinttypes>
#include <cstdio>

using namespace cute;

namespace {

void
print_section(char const* name)
{
  std::printf("\n%s\n", name);
}

void
print_label(char const* name, int indent = 2)
{
  std::printf("%*s%-56s : ", indent, "", name);
}

template <class T>
void
print_kv(char const* name, T const& value, int indent = 2)
{
  print_label(name, indent);
  print(value);
  print("\n");
}

template <class T, size_t N>
void
print_array(char const* name, cute::array<T,N> const& value, int indent = 2)
{
  print_label(name, indent);
  std::printf("{");
  for (size_t i = 0; i < N; ++i) {
    if (i != 0) {
      std::printf(", ");
    }
    if constexpr (sizeof(T) <= sizeof(uint32_t)) {
      std::printf("%" PRIu32, static_cast<uint32_t>(value[i]));
    } else {
      std::printf("%" PRIu64, static_cast<uint64_t>(value[i]));
    }
  }
  std::printf("}\n");
}

} // namespace

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size = Int<1>>
void inspect_make_tma_atom(CopyOp     const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size = {})
{
  auto t1 = make_identity_layout(shape(gtensor));
  auto cta_v_tile = t1.compose(cta_tiler);
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  uint32_t num_multicast = size(cluster_size);

  print_section("inspect_make_tma_atom");
  print_label("copy_op");
  std::printf("SM90_TMA_LOAD\n");
  print_kv("gtensor", gtensor);
  print_kv("gtensor.layout", gtensor.layout());
  print_kv("slayout", slayout);
  print_kv("cta_tiler", cta_tiler);
  print_kv("cluster_size", cluster_size);
  print_kv("num_multicast", num_multicast);
  print_kv("t1", t1);
  print_kv("cta_v_tile", cta_v_tile);

  // detail::make_tma_copy_atom
  // copy_op = SM90_TMA_LOAD{}
  // gtensor = gtensor
  // slayout = slayout
  // num_multicast = size(cluster_size) = _1{}
  // cta_v_map = cta_v_tile
  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  print_section("detail::make_tma_copy_atom");
  print_kv("smem_swizzle", smem_swizzle);
  print_kv("smem_layout", smem_layout);

  // detail::construct_tma_gbasis
  // gtensor = gtensor
  // slayout = slayout
  // cta_v_map = cta_v_tile
  auto inv_smem_layout = right_inverse(smem_layout);
  auto t2 = composition(cta_v_tile, inv_smem_layout);
  auto sidx2gmode_full = coalesce(t2);
  auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);
  auto tile_gstride = recast<TmaType>(gtensor.compose(sidx2gmode)).layout();
  auto tma_gstride  = detail::coalesce_256(tile_gstride);
  auto gbasis = t1;
  auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);
  auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));
  auto t3 = make_layout(wrap(shape(tma_gstride)));
  auto tma_gbasis_tile = tile_gbasis.compose(t3);
  Tensor gtensor_T = recast<TmaType>(gtensor);

  auto t4 = flatten(shape (gtensor_T));
  auto t5 = flatten(stride(gtensor_T));
  auto t6 = flatten(stride(gbasis));
  auto tile_gbasis_remaining_stride = filter_tuple(t4, t5, t6,
                                                   [&](auto s, auto d, auto e)
  {
    if constexpr (is_constant<1, decltype(s)>::value || is_constant<0, decltype(d)>::value) {
      return cute::tuple<>{};          // If size-1 or stride-0, then don't append
    } else {
      using E = decltype(e);
      auto has_e = any_of(flatten(stride(tma_gbasis_tile)), [] (auto tb) { return tb == E{}; });
      if constexpr (decltype(has_e)::value) {
        return cute::tuple<>{};        // If d was found, then don't append
      } else {
        return cute::tuple<E>(e);      // Else, this is missing so append
      }
    }
  });

  auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
  auto t7 = wrap( shape(tma_gbasis_tile));
  auto t8 = wrap(tile_gbasis_remaining_shape );
  auto t9 = wrap(stride(tma_gbasis_tile));
  auto t10 = wrap(tile_gbasis_remaining_stride);
  auto t11 = tuple_cat(t7, t8);
  auto t12 = tuple_cat(t9, t10);
  auto tma_gbasis_full = make_layout(t11, t12);

  auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);

  print_section("detail::construct_tma_gbasis");
  print_kv("inv_smem_layout", inv_smem_layout);
  print_kv("t2 = composition(cta_v_tile, inv_smem_layout)", t2);
  print_kv("sidx2gmode_full", sidx2gmode_full);
  print_kv("smem_rank", smem_rank);
  print_kv("sidx2gmode", sidx2gmode);
  print_kv("tile_gstride", tile_gstride);
  print_kv("tma_gstride", tma_gstride);
  print_kv("gbasis", gbasis);
  print_kv("tile_gbasis_tmp", tile_gbasis_tmp);
  print_kv("tile_gbasis", tile_gbasis);
  print_kv("t3 = make_layout(wrap(shape(tma_gstride)))", t3);
  print_kv("tma_gbasis_tile", tma_gbasis_tile);
  print_kv("gtensor_T", gtensor_T);
  print_kv("gtensor_T.layout", gtensor_T.layout());
  print_kv("t4 = flatten(shape(gtensor_T))", t4);
  print_kv("t5 = flatten(stride(gtensor_T))", t5);
  print_kv("t6 = flatten(stride(gbasis))", t6);
  print_kv("tile_gbasis_remaining_stride", tile_gbasis_remaining_stride);
  print_kv("tile_gbasis_remaining_shape", tile_gbasis_remaining_shape);
  print_kv("t7 = wrap(shape(tma_gbasis_tile))", t7);
  print_kv("t8 = wrap(tile_gbasis_remaining_shape)", t8);
  print_kv("t9 = wrap(stride(tma_gbasis_tile))", t9);
  print_kv("t10 = wrap(tile_gbasis_remaining_stride)", t10);
  print_kv("t11 = tuple_cat(t7, t8)", t11);
  print_kv("t12 = tuple_cat(t9, t10)", t12);
  print_kv("tma_gbasis_full", tma_gbasis_full);
  print_kv("tma_gbasis", tma_gbasis);

  // detail::make_tma_copy_desc
  // gtensor = gtensor
  // tma_gbasis = tma_gbasis
  // swizzle = smem_swizzle
  // num_multicast = size(cluster_size) = _1{}
  constexpr int tma_dim = decltype(rank(tma_gbasis))::value;
  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();
  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
  detail::fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);
  cute::array<uint64_t, 5> gmem_prob_stride_elem = gmem_prob_stride;
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<TmaType>) / 8;
  }
  cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
  cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};
  for_each(make_seq<tma_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(tma_gbasis);
  });
  for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
    assert(smem_box_shape[i] % multicast == 0 || multicast % smem_box_shape[i] == 0);
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }

  print_section("detail::make_tma_copy_desc");
  print_kv("tma_dim", tma_dim);
  print_label("gmem_address");
  std::printf("%p\n", gmem_address);
  print_kv("gmem_layout", gmem_layout);
  print_array("gmem_prob_shape", gmem_prob_shape);
  print_array("gmem_prob_stride (elements)", gmem_prob_stride_elem);
  print_array("gmem_prob_stride (bytes)", gmem_prob_stride);
  print_array("smem_box_shape", smem_box_shape);
  print_array("smem_box_stride", smem_box_stride);

  print_section("cuTensorMapEncodeTiled inputs");
  print_label("tensorMap");
  std::printf("tma_desc\n");
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<TmaType>();
  CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  TMA::SmemSwizzleBits    swizzle_bits    = detail::get_tma_swizzle_bits(smem_swizzle);
  TMA::SmemSwizzleBase    swizzle_base    = detail::get_tma_swizzle_base(smem_swizzle);
  CUtensorMapSwizzle      tma_swizzle     = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);

  print_label("tensorDataType");
  std::printf("%d\n", static_cast<int>(tma_format));
  print_kv("tensorRank", tma_dim);
  print_label("globalAddress");
  std::printf("%p\n", gmem_address);
  print_array("globalDim", gmem_prob_shape);
  print_label("globalStrides");
  std::printf("gmem_prob_stride.data() + 1 = {");
  for (size_t i = 1; i < gmem_prob_stride.size(); ++i) {
    if (i != 1) {
      std::printf(", ");
    }
    std::printf("%" PRIu64, static_cast<uint64_t>(gmem_prob_stride[i]));
  }
  std::printf("}\n");
  print_array("boxDim", smem_box_shape);
  print_array("elementStrides", smem_box_stride);
  print_label("interleave");
  std::printf("%d\n", static_cast<int>(tma_interleave));
  print_label("swizzle_bits");
  std::printf("%d\n", static_cast<int>(swizzle_bits));
  print_label("swizzle_base");
  std::printf("%d\n", static_cast<int>(swizzle_base));
  print_label("swizzle");
  std::printf("%d\n", static_cast<int>(tma_swizzle));
  print_label("l2Promotion");
  std::printf("%d\n", static_cast<int>(tma_l2Promotion));
  print_label("oobFill");
  std::printf("%d\n", static_cast<int>(tma_oobFill));
#else
  print_label("CUDA Driver enum inputs");
  std::printf("not available for this compilation mode\n");
#endif

  auto recast_ratio = cute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<             TmaType>{});

  auto aux_gbasis = make_basis_like(shape(gtensor));

  auto gmem_tma_basis_stride = transform_leaf(aux_gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                 // If size-1 or stride-0, then this basis does not contribute.
    } else {
      auto tma_gmem_basis_stride = stride(tma_gbasis);
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) {
        return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; });
      });
      if constexpr (decltype(j == rank(tma_gmem_basis_stride))::value) {
        return Int<0>{};               // Not found in the TMA basis.
      } else
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
        return E<j>{} * scale;
      } else
      if constexpr (decltype(rank<j>(tma_gmem_basis_stride) == Int<1>{})::value) {
        return E<j>{};
      } else {
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<TmaType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;
      }
    }
  });

  print_section("AuxTmaParams inputs");
  print_kv("recast_ratio", recast_ratio);
  print_kv("aux_gbasis", aux_gbasis);
  print_kv("gmem_tma_basis_stride", gmem_tma_basis_stride);

  // CuTe input to cuTensorMapEncodeTiled
  // tensorMap = tma_desc (just created)
  // tensorDataType = TMA::to_CUtensorMapDataType<TmaType>()
  // tensorRank = tma_dim
  // globalAddress = gmem_address
  // globalDim = gmem_prob_shape.data()
  // globalStrides = gmem_prob_stride.data() + 1,  // gmem_prob_stride[0] implicitly 1
  // boxDim = smem_box_shape.data()
  // elementStrides = smem_box_stride.data()
  // interleave = CU_TENSOR_MAP_INTERLEAVE_NONE
  // swizzle = TMA::to_CUtensorMapSwizzle(get_tma_swizzle_bits(smem_swizzle), get_tma_swizzle_base(smem_swizzle))
  // l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
  // oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
}


int main()
{
  int M = 8192;
  // int N = 8192;
  int K = 8192;
  auto dA = make_stride(K, _1{});
  auto bM = _128{};
  auto bK = _128{};
  auto bP = _6{};
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>{}, make_shape(bM, bK, bP));

  // dummy tensors
  Tensor gtensor = make_tensor(counting_iterator<int>(0), make_shape(M, K), dA);
  auto slayout = sA(_,_,0);
  auto cta_tiler = make_shape(bM, bK);
  

  inspect_make_tma_atom(SM90_TMA_LOAD{}, gtensor, slayout, cta_tiler);
  return 0;
}
