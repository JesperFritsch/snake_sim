

def visitable_squares(odd_tiles: int, even_tiles: int, enter_even: bool, leave_even: bool):
    """
    Calculate the number of squares that can be visited based on the number of odd and even tiles,
    and whether the player can enter or leave even tiles.

    :param odd_tiles: Number of odd tiles
    :param even_tiles: Number of even tiles
    :param enter_even: Can enter even tiles
    :param leave_even: Can leave even tiles
    :return: Total number of visitable squares
    """
    diff = even_tiles - odd_tiles
    more_even = diff > 0
    initially_visitable = min(odd_tiles, even_tiles) * 2
    initially_visitable_is_even = initially_visitable % 2 == 0
    if diff == 0:
        if initially_visitable_is_even:
            return initially_visitable - 1 if enter_even == leave_even else initially_visitable
        else:
            return initially_visitable -  if enter_even == leave_even else initially_visitable






even = 22
odd = 22
enter_even = True
leave_even = True


visitable_squares(odd, even, enter_even, leave_even)