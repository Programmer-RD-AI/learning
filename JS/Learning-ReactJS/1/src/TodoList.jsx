import { TodoItem } from "./TodoItem";

/* eslint-disable react/prop-types */
export function TodoList({ todos, deleteTodo, toggleTodo }) {
  return (
    <ul className="list">
      {todos.length === 0 && "No Todos"}
      {todos.map((todo) => {
        return (
          <TodoItem
            // id={todo.id}
            // completed={todo.completed}
            // title={todo.title}
            key={todo.id}
            {...todo}
            deleteTodo={deleteTodo}
            toggleTodo={toggleTodo}
          />
        );
      })}
    </ul>
  );
}
