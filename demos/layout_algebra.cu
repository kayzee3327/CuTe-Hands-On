#include <cute/tensor.hpp>

int main()
{
  using namespace cute;

  auto A = make_layout(make_shape(_2{},_3{}),make_stride(_3{},_1{}));
  auto B = make_layout(make_shape(_4{},_4{}),make_stride(_1{},_4{}));

  auto lp = logical_product(A, B);
  auto bp = blocked_product(A, B);
  auto rp = raked_product(A, B);

  print("logical_product(A, B): "); print_layout(lp); print("\n");
  print("blocked_product(A, B): "); print_layout(bp); print("\n");
  print("raked_product(A, B)  : "); print_layout(rp); print("\n");

  return 0;
}