#pragma once
#include <cmath>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <string>

#include "LIO-SAM/types.h"

namespace lio_sam {

// ------------------------- Multiplicative Norm ------------------------- //
inline double norm_m2(double p, double i) {
  if (i < 1e-3) {
    i = 1e-3;
  }
  return 1.0 / i;
}
inline double norm_m1(double p, double i) {
  i = std::sqrt(i);
  if (i < 1e-3) {
    i = 1e-3;
  }
  return 1.0 / i;
}

inline double norm0(double p, double i) { return 1.0; }
inline double norm1(double p, double i) {
  i = std::sqrt(i);
  if (i < 1e-3) {
    i = 1e-3;
  }
  return i;
}
inline double norm2(double p, double i) {
  if (i < 1e-3) {
    i = 1e-3;
  }
  return i;
}

// ------------------------- Additive Norm ------------------------- //
inline double add25(double p, double i) {
  if (p < 1e-3) {
    return 0.25 * i;
  } else {
    return 1.0 + 0.25 * i / p;
  }
}

inline double add50(double p, double i) {
  if (p < 1e-3) {
    return 0.50 * i;
  } else {
    return 1.0 + 0.50 * i / p;
  }
}

inline double add75(double p, double i) {
  if (p < 1e-3) {
    return 0.75 * i;
  } else {
    return 1.0 + 0.75 * i / p;
  }
}

// ------------------------- Experiments ------------------------- //
inline double step(double p, double i) { return i < 10 ? 1.0 : 10.0; }

inline double add_norm12(double p, double i) { return (1.0 + std::sqrt(i)); }

inline double add_norm1(double p, double i) { return (1.0 + i); }

inline double add_norm2(double p, double i) { return (1.0 + i * i); }

// ------------------------- Lookup ------------------------- //
const double INTENSITY_SCALE = 1000.0;

template <double (*F)(double, double)> struct CustomMetric {
  typedef bool is_kdtree_distance;

  typedef float ElementType;
  typedef float ResultType;

  template <typename Iterator1, typename Iterator2>
  float operator()(Iterator1 a, Iterator2 b, size_t size,
                   float /*worst_dist*/ = -1) const {
    ResultType result = ResultType();
    ResultType diff;
    diff = *a++ - *b++;
    result += diff * diff;
    diff = *a++ - *b++;
    result += diff * diff;
    diff = *a++ - *b++;
    result += diff * diff;

    float i = INTENSITY_SCALE * abs(*a++ - *b++);
    float scale = F(result, i);

    return result * scale;
  }

  template <typename U, typename V>
  inline ResultType accum_dist(const U &a, const V &b, int) const {
    return (a - b) * (a - b);
  }
};

class IntensityRepresentation
    : public pcl::PointRepresentation<pcl::PointXYZI> {
  using pcl::PointRepresentation<pcl::PointXYZI>::nr_dimensions_;

public:
  IntensityRepresentation() {
    nr_dimensions_ = 4;
    trivial_ = true;
  }

  virtual void copyToFloatArray(const pcl::PointXYZI &p, float *out) const {
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    // TODO: When running with zeros here things work fine - breaks when
    // actually inputting intensity
    // out[3] = 0.0;
    // TODO: Damping by 100 seems to help w/ scale issues - may be good enough
    // for now.
    out[3] = p.intensity / INTENSITY_SCALE;
  }
};

inline pcl::KdTree<PointType>::Ptr setup_kdtree(std::string name) {
  // clang-format off
  if (name == "norm_m2") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<norm_m2>>());
  if (name == "norm_m1") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<norm_m1>>());
  if (name == "norm0") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<norm0>>());
  if (name == "norm1") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<norm1>>());
  if (name == "norm2") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<norm2>>());

  if (name == "add25") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add25>>());
  if (name == "add50") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add50>>());
  if (name == "add75") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add75>>());

  if (name == "step") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<step>>());
  if (name == "add_norm12") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add_norm12>>());
  if (name == "add_norm1") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add_norm1>>());
  if (name == "add_norm2") return pcl::KdTree<PointType>::Ptr(new pcl::KdTreeFLANN<PointType, CustomMetric<add_norm2>>());
  // clang-format on
  throw "Couldn't find metric";
}

} // namespace lio_sam