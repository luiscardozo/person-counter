const WS_HOST = 'localhost';
const MQTT_PORT = 3002;
const FFMPG_PORT = 3004;

export const SETTINGS = {
  CAMERA_FEED_SERVER: "http://" + WS_HOST + ':' + FFMPG_PORT,
  CAMERA_FEED_WIDTH: 852,
  MAX_POINTS: 10,
  SLICE_LENGTH: -10,
};

export const LABELS = {
  START_TEXT: "Click me! ",
  END_TEXT: "The count is now: ",
};

export const HTTP = {
  CAMERA_FEED: `${SETTINGS.CAMERA_FEED_SERVER}/facstream.mjpeg`, // POST
};

export const MQTT = {
  MQTT_SERVER: "ws://" + WS_HOST + ":" + MQTT_PORT,
  TOPICS: {
    PERSON: "person", // how many people did we see
    DURATION: "person/duration", // how long were they on frame
  },
};
