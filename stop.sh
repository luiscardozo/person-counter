#!/bin/bash

SERVER_PID=`ps aux | grep 'node server.js' | grep -v grep | awk '{print $2}'`
if [ ! -z ${SERVER_PID} ]; then
    echo "Closing MQTT Server"
    kill -INT $SERVER_PID
fi

UI_PID=`ps aux | grep 'node.*webservice\/ui.*' | grep -v "NODE_ENV" | awk '{print $2}'`
if [ ! -z ${UI_PID} ]; then
    echo "Closing UI"
    kill -INT $UI_PID
fi

FF_PID=`ps aux | grep 'ffserver' | grep -v grep | awk '{print $2}'`
if [ ! -z ${FF_PID} ]; then
    echo "Closing FFMPEG"
    kill -INT $FF_PID
fi
