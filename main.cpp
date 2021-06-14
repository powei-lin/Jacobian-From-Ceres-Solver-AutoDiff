#include <iostream>

#include <ceres/tiny_solver_autodiff_function.h>

struct Functor {
  template<typename T>
  bool operator()(const T* const parameters, T* residuals) const {
    const T& x = parameters[0];
    const T& y = parameters[1];
    const T& z = parameters[2];
    residuals[0] = x + T(2)*y + T(4)*z;
    residuals[1] = y * z;
    return true;
  }
};

using namespace std;

int main() {
  //Tiny...<Functor, numOfOutput, numOfInput, Scalar>
  using AutoDiffFunction = ceres::TinySolverAutoDiffFunction<Functor, 2, 3, float>;

  Functor my_functor;
  AutoDiffFunction func(my_functor);
  Eigen::Vector3f input(1, 2, 3);
  Eigen::Vector2f output;
  Eigen::Matrix<float, 2, 3> jacobian_matrix;

  func(input.data(), output.data(), jacobian_matrix.data());
  cout << "\nf(x, y, z) = [x+2y+4z, yz]" << endl << endl;
  cout << "input:\n" << input << endl << endl;
  cout << "output:\n" << output << endl << endl;
  cout << "jacobian:\n";
  cout << jacobian_matrix << endl;

  return 0;
}