# Error Handling

Rust distiguishes between two types of errors: recoverable and unrecoverable.

Unrecoverable errors are those that a program should not try to recover from. For example, if a program tries to open a file that does not exist, it should not try to recover from this error. Instead, it should panic and terminate the program. 

Recoverable errors are those that a program can try to recover from. For example, if a program tries to open a file that does not exist, it can try to create the file instead. It's handled using the `Result` type. The `Result` type is an enum that can be either `Ok` or `Err`. The `Ok` variant contains the value that was returned, while the `Err` variant contains the error that occurred.
