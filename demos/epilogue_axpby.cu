#include <cute/tensor.hpp>

int main(int argc, char* argv[])
{
  using namespace cute;
  const int m = 5120;
  const int n = 5120;
  const int k = 5120;
  float* dummy = nullptr;
  auto shape_MNK = make_shape(m,n,k);
  auto dC = make_stride(n, _1{});
  auto cta_tiler = make_shape(_128{}, _128{}, _8{});
  auto coord = make_coord(0, 0, _);
  Tensor mC = make_tensor(make_gmem_ptr(dummy), select<0, 1>(shape_MNK), dC);
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));
  TiledMMA mma = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                Layout<Shape<_16, _16, _1>>{},
                                Tile<Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                     Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                     _1>{});
  // Copy_Atom<UniversalCopy<uint128_t>, float> copy_epi_atom;
  // TiledCopy copy_epi = make_tiled_copy_C(copy_epi_atom, mma);
  // print_latex(copy_epi);
  ThrMMA thr_mma = mma.get_slice(0);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = make_fragment_like(tCgC);
  Tensor tCrD = make_fragment_like(tCrC);
  auto copy_epi =
  make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                  Layout<Shape<_16, _16>, Stride<_1, _16>>{},
                  Layout<Shape<_4, _4>, Stride<_4, _1>>{});
  ThrCopy thr_copy_epi = copy_epi.get_slice(0);
  Tensor tXrD = thr_copy_epi.retile_S(tCrD);
  Tensor tXgD = thr_copy_epi.partition_D(gC);
  print_latex(copy_epi);
}