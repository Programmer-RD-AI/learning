# Further Match Expressions

## `if let`

Handles one pattern and ignores the rest, useful for only 1 specific variant.

## `while let`

Loops as long as a pattern continues to match, simplifies cases like extracting values from an `Option` until it's empty.

## `let ... else`

Allows a refutable pattern in a let binding where failuire jumps to the else block which must diverge.

