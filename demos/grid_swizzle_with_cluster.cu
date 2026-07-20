#include <cute/tensor.hpp>

template<uint32_t GM, uint32_t GN, class ClusterShape>
auto make_swizzled_tile_layout(
  uint32_t tiles_m,
  uint32_t tiles_n,
  ClusterShape
) {
  using namespace cute;

  static_assert(rank_v<ClusterShape> == 3,
                "ClusterShape must have M, N, and K modes");

  constexpr uint32_t CM = size<0>(ClusterShape{});
  constexpr uint32_t CN = size<1>(ClusterShape{});
  constexpr uint32_t CK = size<2>(ClusterShape{});
   print("CM: "); print(CM); print("\n");
   print("CN: "); print(CN); print("\n");
   print("CK: "); print(CK); print("\n");
   print("GM: "); print(GM); print("\n");
   print("GN: "); print(GN); print("\n");

  static_assert(CK == 1, "Grid swizzling only supports ClusterShape K == 1");
  static_assert(GM > 0 && GN > 0, "Swizzle-group dimensions must be positive");
  static_assert(GM % CM == 0,
                "The M swizzle group must contain complete clusters");
  static_assert(GN % CN == 0,
                "The N swizzle group must contain complete clusters");

  constexpr uint32_t cluster_group_m = GM / CM; // view CTA groups as cluster groups
  constexpr uint32_t cluster_group_n = GN / CN;
  static_assert(cluster_group_m == cluster_group_n, "Cluster swizzle shape should be square."); // ?
  print("cluster_group_m: "); print(cluster_group_m); print("\n");
  print("cluster_group_n: "); print(cluster_group_n); print("\n");
  uint32_t clusters_m = tiles_m / CM; // clusters in the grid
  uint32_t clusters_n = tiles_n / CN;
  print("clusters_m: "); print(clusters_m); print("\n");
  print("clusters_n: "); print(clusters_n); print("\n");

  // Preconditions for this initial version:
  // tiles_m % GM == 0 && tiles_n % GN == 0

  // Physical cluster BID digits:
  //   (local_cluster_m, local_cluster_n, group_m, group_n)
  //
  // output:
  //   natural logical cluster index cluster_m + clusters_m * cluster_n
  //
  // Keep this traversal rank-1 so blocked_product treats each CM x CN CTA
  // cluster as one indivisible block.
  auto s = make_shape (Int<cluster_group_m>{},
                       Int<cluster_group_n>{},
                       clusters_m / cluster_group_m,
                       clusters_n / cluster_group_n); // size(s) == clusters_m * clusters_n
  auto d = make_stride(_1{},
                       clusters_m,
                       Int<cluster_group_m>{},
                       clusters_m * cluster_group_n); // cluster groups as rectangle tiles
  auto cluster_bid_to_logical_linear = make_layout(make_tuple(s), 
                                                   make_tuple(d));
  // auto cluster_bid_to_logical_linear = make_layout(s, d);
  print(cluster_bid_to_logical_linear); print("\n");
  auto cta_in_cluster_layout =
      Layout<Shape<Int<CM>, Int<CN>>>{}; // inside cluster group
  print(cta_in_cluster_layout); print("\n");
  auto bid_to_logical_linear =
      blocked_product(cta_in_cluster_layout,
                      cluster_bid_to_logical_linear);
  print(bid_to_logical_linear); print("\n");

  // Natural logical linear index -> (logical_m, logical_n)
  auto logical_coord = make_identity_layout(make_shape(tiles_m, tiles_n));
  print(logical_coord); print("\n");
  auto res = composition(logical_coord, bid_to_logical_linear);
  print(res); print("\n");

  return res;
}

int main()
{
  auto res = make_swizzled_tile_layout<8,4>(
    64,64,cute::Shape<cute::_2,cute::_1,cute::_1>{}
  );
}