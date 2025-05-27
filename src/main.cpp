#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include "gameOfLife/cpu.hpp"
#include "gameOfLife/cuda.hpp"
#include "gameOfLife/interface.hpp"
#include "gameOfLife/opencl.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <vector>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
std::vector<float> vertices;
unsigned int VBO, VAO, EBO;
std::map<int, bool> key_press;

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 ourColor;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    ourColor = aColor;
}
)";
const char *fragmentShaderSource = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(ourColor, 1.0);
}
)";

int main(int argc, char **argv) {
  bool cuda = false;
  bool opencl = false;
  bool cpu = true;
  for (int i = 0; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--cuda") {
      cuda = true;
    } else if (arg == "--opencl") {
      opencl = true;
    }
  }
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Game Of Life", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // build and compile our shader program
  // ------------------------------------
  // vertex shader
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  // check for shader compile errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  // fragment shader
  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  // check for shader compile errors
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  // link shaders
  unsigned int shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  // check for linking errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
  }
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  int N = 200;
  int M = 300;
  std::vector<std::vector<int>> grid(N, std::vector<int>(M, 1));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      grid[i][j] = rand() % 2 == 0;
    }
  }
  std::unique_ptr<GameOfLifeInterface> gol;
  if (cpu)
    gol = std::make_unique<GameOfLifeCPU>(grid);
  if (cuda)
    gol = std::make_unique<GameOfLifeCuda>(grid);
  if (opencl)
    gol = std::make_unique<GameOfLifeOpenCL>(grid);
  glfwSetWindowUserPointer(window, gol.get());

  // fraction of step
  float gap_frac = 0.20f;
  float step = 2.0f / ((float)std::max(N, M) + 1);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      float pos_x = (N >= M ? -1.0f : (N - M) * step) + step * (i + 1);
      float pos_y = (M >= N ? -1.0f : (M - N) * step) + step * (j + 1);
      float gap = step / 2 - step * gap_frac;
      float left = pos_x - gap;
      float right = pos_x + gap;
      float top = pos_y + gap;
      float bottom = pos_y - gap;

      float r = grid[i][j] ? 1.0f : 0.3f;
      float g = r;
      float b = r;
      // clang-format off
      vertices.insert(vertices.end(), {
          right, top, 0.0f, r, g, b,
          right, bottom, 0.0f, r, g, b,
          left, bottom, 0.0f, r, g, b,
      });
    vertices.insert(vertices.end(), {
        left, top, 0.0f, r, g, b,
        left, bottom, 0.0f, r, g, b,
        right, top, 0.0f, r, g, b,
    });
      // clang-format on
    }
  }
  unsigned int indices[] = {
      // note that we start from 0!
      0, 1, 3, // first Triangle
      1, 2, 3  // second Triangle
  };
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  // bind the Vertex Array Object first, then bind and set vertex buffer(s), and
  // then configure vertex attributes(s).
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
  //              GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // note that this is allowed, the call to glVertexAttribPointer registered VBO
  // as the vertex attribute's bound vertex buffer object so afterwards we can
  // safely unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // remember: do NOT unbind the EBO while a VAO is active as the bound element
  // buffer object IS stored in the VAO; keep the EBO bound.
  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // You can unbind the VAO afterwards so other VAO calls won't accidentally
  // modify this VAO, but this rarely happens. Modifying other VAOs requires a
  // call to glBindVertexArray anyways so we generally don't unbind VAOs (nor
  // VBOs) when it's not directly necessary.
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw our first triangle
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it
                            // every time, but we'll do so to keep things a bit more organized
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 6);
    gol->tick();
    auto curr_grid = gol->get_grid();
    for (int i = 0; i < curr_grid.size(); i++) {
      for (int j = 0; j < curr_grid[0].size(); j++) {
        float v = curr_grid[i][j] ? 1.0f : 0.3f;
        int base_index = (i * curr_grid[0].size() + j) * 6 * 6;
        for (int vertex = base_index; vertex < base_index + 6 * 6; vertex += 6) {
          vertices[vertex + 3] = v;
          vertices[vertex + 4] = v;
          vertices[vertex + 5] = v;
        }
      }
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    // glBindVertexArray(0); // no need to unbind it every time

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteProgram(shaderProgram);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !key_press[GLFW_KEY_SPACE]) {
    key_press[GLFW_KEY_SPACE] = true;
    GameOfLifeCPU *pgol = static_cast<GameOfLifeCPU *>(glfwGetWindowUserPointer(window));
    pgol->tick();
    auto grid = pgol->get_grid();
    for (int i = 0; i < grid.size(); i++) {
      for (int j = 0; j < grid[0].size(); j++) {
        float v = grid[i][j] ? 1.0f : 0.3f;
        int base_index = (i * grid[0].size() + j) * 6 * 6;
        for (int vertex = base_index; vertex < base_index + 6 * 6; vertex += 6) {
          for (int index = vertex + 3; index < vertex + 6; index++) {
            vertices[index] = v;
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, index * sizeof(float), sizeof(float), &vertices[index]);
          }
        }
      }
    }
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
    key_press[GLFW_KEY_SPACE] = false;
  }
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  int new_width = std::min(width, height);
  int new_height = new_width;
  int x = 0;
  int y = 0;
  if (width > height) {
    x = (width - height) / 2;
  }
  if (height > width) {
    y = (height - width) / 2;
  }
  glViewport(x, y, new_width, new_height);
}
