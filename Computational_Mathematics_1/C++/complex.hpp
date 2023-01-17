#ifndef COMPLEX_HPP
#define COMPLEX_HPP

namespace math
{
    class Complex
    {

    private:
        double _real;
        double _imag;

    public:
        Complex();
        Complex(const double value);
        Complex(const double real, const double imag);
        Complex(const Complex &copy);

        double real() const;
        double imag() const;
        double angle() const;
        double length() const;
        Complex conjugate() const;

        Complex operator+(const Complex &rhs) const;
        Complex operator+(const double scale) const;
        Complex operator-(const Complex &rhs) const;
        Complex operator-(const double scale) const;
        Complex operator*(const Complex &rhs) const;
        Complex operator*(const double scale) const;
        Complex operator/(const Complex &rhs) const;
        Complex operator/(const double scale) const;

        Complex &operator+=(const Complex &rhs);
        Complex &operator+=(const double scale);
        Complex &operator-=(const Complex &rhs);
        Complex &operator-=(const double scale);
        Complex &operator*=(const Complex &rhs);
        Complex &operator*=(const double scale);
        Complex &operator/=(const Complex &rhs);
        Complex &operator/=(const double scale);

        Complex &operator=(const Complex &rhs);
    };
    extern const Complex I;
}


#endif