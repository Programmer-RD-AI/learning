// eslint-disable-next-line react/prop-types
export function TodoItem({ completed, id, title, toggleTodo, deleteTodo }) {
  return (
    <li key={id}>
      <label>
        <input type="checkbox" onChange={() => toggleTodo(id, completed)} />
        {title}
      </label>
      <button onClick={() => deleteTodo(id)} className="btn btn-danger">
        Delete
      </button>
    </li>
  );
}
