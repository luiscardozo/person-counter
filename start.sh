#!/bin/bash

#My local installation, as the ffmpeg installed via apt does not have ffserver anymore
FFSERVER=/opt/bin/ffserver/bin/ffserver
FFMPEG=/opt/bin/ffserver/bin/ffmpeg

UI_SERVER_URL=http://localhost:3000

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#prepare the logs
LOGDIR=$SCRIPT_DIR/logs
mkdir -p $LOGDIR

echo "Running Mosca Server..."
cd $SCRIPT_DIR/webservice/server/node-server/
exec node server.js > $LOGDIR/mosca.log 2> $LOGDIR/mosca_err.log &
sleep 2

echo "Running the UI server..."
cd $SCRIPT_DIR/webservice/ui/
exec npm run dev > $LOGDIR/ui.log 2> $LOGDIR/ui_err.log &
sleep 2

echo "Running the FFMPEG server..."
cd $SCRIPT_DIR/ffmpeg
exec $FFSERVER -f server.conf > $LOGDIR/ffserver.log 2> $LOGDIR/ffserver_err.log &
sleep 2

# open the UI in a browser
echo "Opening the browser..."
xdg-open http://localhost:3000

echo "... and running the python script"
cd $SCRIPT_DIR
source env/bin/activate

VIDEO_SIZE=768x432
python3 main.py | $FFMPEG -v warning -f rawvideo -pixel_format bgr24 -video_size $VIDEO_SIZE -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
