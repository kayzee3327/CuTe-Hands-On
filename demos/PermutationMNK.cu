#include <cute/tensor.hpp>

// const int M = 256;
// const int N = 256;
// const int K = 256;

int main(int argc, char* argv[])
{
  using namespace cute;

  // This experiment is to observe why it is necessary apply PermutationMNK in TiledMMA
  //  and to explore potential alternative solution

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <integer_argument>\n";
    return 1; // Return an error code
  }
  int choice = std::stoi(argv[1]);
  // 1. Let's see what CuTe does in v1 
  // TiledCopy tcp = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
  //                                 Layout<Shape<_32,_8>>{},
  //                                 Layout<_4,_1>{});
  TiledMMA mma1 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{});
  // auto thr_mma1 = mma1.get_slice(0);
  // float* dummy = nullptr;
  // auto sA = make_tensor(make_smem_ptr(dummy),
  //                       make_layout(make_shape(_128{}, _8{})));
  // auto tCsA = thr_mma1.partition_A(sA);
  // auto tCrA = thr_mma1.make_fragment_A(tCsA);

  // print("tCsA      : "); print(tCsA.layout()); print("\n");
  // print("tCrA      : "); print(tCrA.layout()); print("\n");
  // print("common vec: "); print(max_common_vector(tCsA(_,_,0), tCrA(_,_,0))); print("\n");
  // print_latex(mma1.get_layoutA_TV());
  // print(tcp);

  // 2. Let's see what PermutationMNK can do
  // this should be identical to mma1
  TiledMMA mma2 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<_16, _16, _1>{});
  
  // try to tile M mode 
  TiledMMA mma3 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _2>>, 
                                      _16, 
                                      _1>{});

  // try to rearrange M mode 
  TiledMMA mma4 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>, 
                                      _16, 
                                      _1>{});

  // merge thread tile (change _2 to actual thread tile size)
  TiledMMA mma5 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                      Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                      _1>{});
  
  // make it LDS.128 friendly for 8*8 thread tile
  TiledMMA mma6 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, Shape<_4, _2>>, Stride<_4, Stride<_1, _64>>>,
                                      Layout<Shape<_16, Shape<_4, _2>>, Stride<_4, Stride<_1, _64>>>,
                                      _1>{});

  if (choice == 1)
  {
    print_latex(mma1);
  }
  else if (choice == 2)
  {
    print_latex(mma2);
  }
  else if (choice == 3)
  {
    print_latex(mma3);
  }
  else if (choice == 4)
  {
    print_latex(mma4);
  }
  else if (choice == 5)
  {
    print_latex(mma5);
  }
  else if (choice == 6)
  {
    print_latex(mma6);
  }
  
  
  
  return 0;

}
