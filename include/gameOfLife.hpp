#include <vector>

class GameOfLife {
private:
  std::vector<std::vector<bool>> grid;
public:
  GameOfLife(std::vector<std::vector<bool>> &grid);
  void tick();
};
