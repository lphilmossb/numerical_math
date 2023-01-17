#include "complex.hpp"
#include <cmath>

using math::Complex;
using math::I;


const Complex I = Complex(0.0, 1.0);

Complex::Complex(): _real(0.0), _imag(0.0) {}
Complex::Complex(const double value): _real(value), _imag(value) {}
Complex::Complex(const double real, const double imag): _real(real), _imag(imag) {}

double Complex::real() const {
    return _real;
}

double Complex::imag() const {
    return _imag;
}

double Complex::angle() const {
    return std::atan2(_imag, _real);
}

double Complex::length() const {
    return std::sqrt(_real * _real + _imag * _imag);
}

Complex Complex::conjugate() const {
    return Complex(_real, -1.0 * _imag);
}


Complex Complex::operator+(const Complex& rhs) const {
    return Complex(_real + rhs._real, _imag + rhs._imag);
}

Complex Complex::operator-(const Complex& rhs) const {
    return Complex(_real - rhs._real, _imag - rhs._imag);
}

Complex Complex::operator*(const Complex& rhs) const {
    return Complex(_real*rhs._real - _imag * rhs._imag, _real * rhs._imag + _imag * rhs._real);
}

Complex Complex::operator/(const Complex& rhs) const {
    double length = rhs.length();
    length *= length;
    Complex inv = Complex(rhs._real / length, -1.0 * rhs._imag / length);
    return *this * inv;
}
