#include <string>
#include <stdexcept>

double
stod_strict(const std::string& str)
{
  size_t i = 0;
  double v = std::stod(str, &i);
  for (size_t n = str.length(); i < n; i++) {
    if (!std::isspace(str[i]))
      throw std::invalid_argument("stod_strict: no conversion");
  }
  return v;
}

float
stof_strict(const std::string& str)
{
  size_t i = 0;
  float v = std::stof(str, &i);
  for (size_t n = str.length(); i < n; i++) {
    if (!std::isspace(str[i]))
      throw std::invalid_argument("stof_strict: no conversion");
  }
  return v;
}

int
stoi_strict(const std::string& str)
{
  size_t i = 0;
  int v = std::stoi(str, &i);
  for (size_t n = str.length(); i < n; i++) {
    if (!std::isspace(str[i]))
      throw std::invalid_argument("stod_strict: no conversion");
  }
  return v;
}
