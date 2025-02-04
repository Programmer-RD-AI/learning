import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { Dimensions, Image, StyleSheet, Text, View,ActivityIndicator } from 'react-native';
import Loading from './app/screen/Loading';
const dimensions = Dimensions.get("screen");

export default function App() {
  return (
    <View style={styles.container}>
      <Image style={{position:"absolute",left:0,top:dimensions['height']-500}} source={require('./app/assets/small_thing_1.png')}/>
      <Image style={{position:"absolute",right:0,top:dimensions['height']-250}} source={require('./app/assets/small_thing_2.png')}/>
     <Loading />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFBD00",
    alignItems: 'center',
    justifyContent: 'center',
  },
});
