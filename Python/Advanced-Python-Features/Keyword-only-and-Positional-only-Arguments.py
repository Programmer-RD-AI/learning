#   KW+POS | KW ONLY

#       vv | vv


def foo(a, *, b): ...


# == ALLOWED ==

foo(a=1, b=2)  # All keyword

foo(1, b=2)  # Half positional, half keyword


# == NOT ALLOWED ==

foo(1, 2)  # Cannot use positional for keyword-only parameter

#      ^
# POS ONLY | KW POS

#       vv | vv


def bar(a, /, b): ...


# == ALLOWED ==

bar(1, 2)  # All positional

bar(1, b=2)  # Half positional, half keyword


# == NOT ALLOWED ==

bar(a=1, b=2)  # Cannot use keyword for positional-only parameter

#   ^
