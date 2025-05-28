#include "interface.hpp"
#include <CL/opencl.hpp>

class GameOfLifeOpenCL : public GameOfLifeInterface {
private:
  std::vector<std::vector<int>> grid;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Buffer bufferIn;
  cl::Buffer bufferOut;
  cl::Kernel gol_kernel;
  size_t n, m;
  std::vector<int> bufferInHost;
  std::vector<int> bufferOutHost;
  int workgroup_x;
  int workgroup_y;

public:
  GameOfLifeOpenCL(std::vector<std::vector<int>> &grid, int workgroup_x, int workgroup_y);
  void tick();
  const std::vector<std::vector<int>> get_grid() const;
};
