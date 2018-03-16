import argparse,os,sys
import tensorflow as tf
import numpy as np
from tool import tf_resize_images, load_graph, preprocess_image
from tool import tf_web_resize_images
#from tensorflow.python.keras._impl.keras import backend as K
import http.server
import socketserver
import csv

imagesize = [299,299]
FLAGS = None

#session_conf = tf.ConfigProto()
session_conf = tf.ConfigProto(
    device_count={'CPU' : 8, 'GPU' : 0},
    allow_soft_placement     = False,
    log_device_placement     = False,
)
session_conf.gpu_options.allow_growth = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

nametag_SING = []
with open('./database/S208.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        nametag_SING.append(row[0])

nametag_WEST = []
with open('./database/W101CH.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        nametag_WEST.append(row[0])
nametag_WEST = nametag_WEST

carbs_SING = []
with open('./database/S208carbs.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        carbs_SING.append(float(row[0]))

carbs_WEST = []
with open('./database/W101carbs.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        carbs_WEST.append(float(row[0]))

class MyHandler(http.server.SimpleHTTPRequestHandler):
    # Perform image classification
    def myproc(self, image_data):
        imgdata = tf_web_resize_images(image_data,imagesize)
        Imgdata = np.expand_dims(imgdata,0)
        Imgdata = preprocess_image(Imgdata)

        with tf.Session(graph=graph,config=session_conf) as sess:
            #K.set_session(sess)
            y_out, top_out = sess.run([y,Top], feed_dict={
                x: Imgdata, phase: 0
            })
        with tf.Session(graph=graph2,config=session_conf) as sess:
            #K.set_session(sess)
            y_out2, top_out2 = sess.run([y2,Top2], feed_dict={
                x2: Imgdata, phase2: 0
            })

        #with tf.Session(graph=graph3,config=session_conf) as sess:
            #K.set_session(sess)
        #    y_out3, top_out3 = sess.run([y3,Top3], feed_dict={
        #        x3: Imgdata, phase3: 0
        #    })

        buf = ""
        dictmerge = {}
        for toptag,topvalue in zip(np.squeeze(top_out.indices),np.squeeze(top_out.values)):
            human_string = nametag_WEST[toptag]
            carbs = carbs_WEST[toptag]
            dictmerge.update({human_string:(topvalue,carbs)})
        for toptag,topvalue in zip(np.squeeze(top_out2.indices),np.squeeze(top_out2.values)):
            human_string = nametag_SING[toptag]
            carbs = carbs_SING[toptag]
            dictmerge.update({human_string:(topvalue,carbs)})
        #for toptag,topvalue in zip(np.squeeze(top_out3.indices),np.squeeze(top_out3.values)):
        #    human_string = nametag_WEST[toptag]
        #    carbs = carbs_WEST[toptag]
        #    dictmerge.update({human_string:(topvalue,carbs)})
        sortrank = sorted(dictmerge.items(), key=lambda value: value[1], reverse=True)
        for ii in range(int(topnum)):
            if buf:
                buf = buf + ","
            buf = buf + ('{"name":"%s", "carbs":%.1f}' % (sortrank[ii][0],sortrank[ii][1][1]))
        return buf
    def do_GET(self):
        # web address is localhost:portnum/input
        if self.path == "/input":
            f = open("./input.html", encoding='utf-8')
            # Read the website as binary and encode it as utf-8
            body = (f.read()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Content-length', len(body))
            self.end_headers()
            self.wfile.write(body)
            return
        # web address is localhost:portnum
        body=b'{"classify_image": "ok"}'    
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        #print(self.data_string)
        body = ('{"classify": [%s]}' % ( self.myproc(self.data_string))).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-length', len(body))
        self.end_headers()
        self.wfile.write(body)

def run_web_server():    
    print("start web server!")
    PORT = 9998
    httpd = socketserver.TCPServer(("", PORT), MyHandler)
    print("serving at port", PORT)
    httpd.serve_forever()

def create_graph(pbfilename,topnum):
    """Creates a graph from saved GraphDef file and returns a saver."""
    graph = load_graph(pbfilename)

    x = graph.get_tensor_by_name('prefix/input_aug:0')
    phase = graph.get_tensor_by_name('prefix/phase:0')
    beforeFC = graph.get_tensor_by_name('prefix/beforeFC:0')
    y = tf.nn.softmax(graph.get_tensor_by_name('prefix/outputnode:0'))
    Top = tf.nn.top_k(y,k=int(topnum))
    return x,y,phase,Top,graph

def run_inference_on_image(imagefiledir,pbfilename,topnum):
    """Runs inference on an image."""
    imgdata = tf_resize_images(imagefiledir,imagesize)
    Imgdata = np.expand_dims(imgdata,0)
    Imgdata = preprocess_image(Imgdata)

    x,y,phase,Top,graph = create_graph(pbfilename,topnum)

    with tf.Session(graph=graph) as sess:
        #K.set_session(sess)
        y_out, top_out = sess.run([y,Top], feed_dict={
            x: Imgdata, phase: 0
        })
    for toptag,topvalue in zip(np.squeeze(top_out.indices),np.squeeze(top_out.values)):
        print('%20s\t, prob = %7.5f' % (nametag_EN[toptag],topvalue))

def main(_):
    run_web_server()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbmodel", type=str, help="Frozen model file to import")
    parser.add_argument("--pbmodel2", type=str, help="Frozen model file to import")
    #parser.add_argument("--pbmodel3", type=str, help="Frozen model file to import")
    parser.add_argument("--image"  , type=str, help="Image file to import")
    parser.add_argument("--top"    , type=str, help="Show top perdictions")
    FLAGS, unparsed = parser.parse_known_args()
    pbmodel  = (FLAGS.pbmodel if FLAGS.pbmodel else 'frozen_WESTV42.pb')
    pbmodel2 = (FLAGS.pbmodel if FLAGS.pbmodel else 'frozen_SINGV4.pb')
    #pbmodel3 = (FLAGS.pbmodel if FLAGS.pbmodel else 'frozen_WESTV4.pb')
    image    = (FLAGS.image   if FLAGS.image   else '../pho.jpg')
    topnum   = (FLAGS.top     if FLAGS.top     else '5')
    x,  y,phase ,Top ,graph  = create_graph(pbmodel,topnum)
    x2,y2,phase2,Top2,graph2 = create_graph(pbmodel2,topnum)
    #x3,y3,phase3,Top3,graph3 = create_graph(pbmodel3,topnum)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


