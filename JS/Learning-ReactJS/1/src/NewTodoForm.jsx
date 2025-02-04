import { useState } from "react";

// eslint-disable-next-line react/prop-types
export function NewTodoForm({ onSubmit }) {
  const [newItem, setNewItem] = useState("");
  function handleSubmit(e) {
    e.preventDefault();
    if (newItem === "") return;
    onSubmit(newItem);
    setNewItem("");
  }
  return (
    <form className="new-item-form" onSubmit={handleSubmit}>
      <div className="form-row">
        <label htmlFor="item">New Item</label>
        <input
          type="text"
          id="item"
          onChange={(val) => {
            setNewItem(val.target.value);
          }}
          value={newItem}
        />
      </div>
      <button className="btn">Add</button>
    </form>
  );
}
