import React from "react";
import { Text, StyleSheet, View, Image, Dimensions } from "react-native";
const dimensions = Dimensions.get("screen");
function Loading({ img_url, text_main, company_name, made_by }) {
  return (
    <>
      <View style={{ flex: 1 }}>
        <Image
          style={styles.image}
          source={{
            uri: "https://cdn.dribbble.com/users/2080921/screenshots/14908381/media/ead82a9b992b214980c75822f89f53dc.png?compress=1&resize=400x300",
            width: 323,
            height: 294,
          }}
        />
        <Text style={styles.mainText}>{"Like Reading Stories ? \nUse"}</Text>
        <Text style={styles.nameText}>{"blog \n stories"}</Text>
      
        
        <Text style={styles.made_by}>{"Made By Ranuga Disansa"}</Text>
      </View>
    </>
  );
}
const styles = StyleSheet.create({
  image: {
    top: 0,
    bottom: dimensions["width"] * 0.68,
    // left: dimensions["height"] * 0.083,
    // right: dimensions["height"] * 0.083,
    borderRadius: 147,
  },
  mainText: {
    fontFamily: "serif",
    fontStyle: "normal",
    fontWeight: "normal",
    fontSize: 25,
    lineHeight: 31,
    alignItems: "center",
    textAlign: "center",
    color: "#464646",
  },
  nameText: {
    fontFamily: "serif",
    fontStyle: "normal",
    fontWeight: "normal",
    fontSize: 50,
    lineHeight: 50,
    alignItems: "center",
    textAlign: "center",
    color: "#464646",
    marginTop: 10,
  },
  made_by: {
    fontFamily: "serif",
    fontStyle: "normal",
    fontWeight: "normal",
    fontSize: 25,
    lineHeight: 50,
    alignItems: "center",
    textAlign: "center",
    color: "#464646",
    position: "absolute",
    bottom: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  design: {
    position: "absolute",
    width: "135px",
    height: "69px",
    left: "82.46px",
    top: "565.84px",
    elevation: 15,
  },
});
export default Loading;
