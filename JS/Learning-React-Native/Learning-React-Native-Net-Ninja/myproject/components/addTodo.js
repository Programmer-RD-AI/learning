import React, { useState } from "react";
import {
  StyleSheet,
  View,
  TextInput,
  TouchableOpacity,
  Button,
} from "react-native";
const AddTodo = ({ submitHandler }) => {
  const [text, setText] = useState(null);
  const ChangeHandler = (value) => {
    setText(value);
  };
  return (
    <View>
      <TextInput
        style={styles.input}
        placeholder="New Todo..."
        onChangeText={ChangeHandler}
        clearButtonMode="always"
        value={text}
      />
      <Button
        onPress={() => {
          submitHandler(text);
          setText("");
        }}
        title="Add Todo"
        color="coral"
      />
    </View>
  );
};
const styles = StyleSheet.create({
  input: {
    marginBottom: 10,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: "#ddd",
  },
});
export default AddTodo;
