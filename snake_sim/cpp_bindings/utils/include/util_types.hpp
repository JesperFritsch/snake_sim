#pragma once
#include <cstddef>      // for size_t
#include <stdexcept>    // for std::out_of_range


struct Coord
{
    int x;
    int y;

    Coord() : x(-1), y(-1) {}

    Coord(int x, int y) : x(x), y(y) {}
    // Overload operator[] to access x and y like an array
    int &operator[](size_t index)
    {
        if (index == 0)
            return x;
        if (index == 1)
            return y;
        throw std::out_of_range("Index out of range");
    }

    const int &operator[](size_t index) const
    {
        if (index == 0)
            return x;
        if (index == 1)
            return y;
        throw std::out_of_range("Index out of range");
    }

    bool operator==(const Coord &other) const
    {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const Coord &other) const
    {
        return x != other.x || y != other.y;
    }

    Coord operator-(const Coord &other) const
    {
        return Coord(x - other.x, y - other.y);
    }

    Coord operator+(const Coord &other) const
    {
        return Coord(x + other.x, y + other.y);
    }

    std::size_t hash() const
    {
        return x * 1000 + y;
    }
};


namespace std
{
    template <>
    struct hash<Coord>
    {
        std::size_t operator()(const Coord &coord) const noexcept
        {
            return coord.hash();
        }
    };
}