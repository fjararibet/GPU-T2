#include <vector>

class GameOfLife {
private:
  std::vector<std::vector<bool>> grid;
public:
  GameOfLife(std::vector<std::vector<bool>> &grid);
  void tick();
  const std::vector<std::vector<bool>> get_grid() const;
};
