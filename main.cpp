#include <chrono>
#include <iostream>

#if defined(MATMUL)
extern "C" float matmul();
#elif defined(VECADD)
extern "C" float vecadd();
#elif defined(CONV)
extern "C" float conv();
#endif

int main(int argc, char *argv[]) {
  std::chrono::time_point<std::chrono::system_clock> startTime =
      std::chrono::system_clock::now();

#if defined(MATMUL)
  float v = matmul();
#elif defined(VECADD)
  float v = vecadd();
#elif defined(CONV)
  float v = conv();
#else
  float v = 0.0f;
#endif

  std::chrono::duration<double> elapsedTime =
      std::chrono::system_clock::now() - startTime;

  std::cout << "op result = " << v << ", time = " << elapsedTime.count()
            << " seconds." << std::endl;
  return 0;
}
