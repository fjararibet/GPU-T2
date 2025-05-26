#include <vector>

class GameOfLifeInterface {
public:
  virtual void tick() = 0;
  virtual const std::vector<std::vector<int>> get_grid() const = 0;
  virtual ~GameOfLifeInterface() {};
};
