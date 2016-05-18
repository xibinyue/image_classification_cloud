import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory,session
from werkzeug import secure_filename

import image_classification
from image_classification import initialize_caffe
import urllib

import sys
caffe_root = '/Users/seanlu/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe



tag_data = {}

net,labels,transformer = initialize_caffe()

# Initialize the Flask application
app = Flask(__name__,static_url_path = "", static_folder = "uploads")

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config.update(dict(
    SECRET_KEY='development key',
))


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/',methods=['GET', 'POST'])
def index():

    print (tag_data)

    if request.method == 'POST':
        if request.form.get("subject", None) == "Upload an image":
            upload_an_image = True
            return render_template('index.html', upload_an_image=upload_an_image)
        
        elif request.form.get("subject", None) == "Search images based on tags":
            search_an_image = True
            return render_template('index.html', search_an_image=search_an_image)
        
        elif request.form.get("subject", None) == "start_search":
            search_an_image = True
            start_search = True

            searched_tag = str(request.form.get("searched_tag", None))
            s_filenames = []

            print (tag_data)

            for i_file, i_tags in tag_data.iteritems():
                print (i_file)
                print (i_tags)
                if (searched_tag in str(i_tags)):
                
                    s_filenames.append(str(i_file))

            if (len(s_filenames) == 0):
                messages = "No images found!"
                return render_template('index.html', search_an_image=search_an_image, messages=messages)
            else:
                #for i in s_filenames:
                #    print (i)
                return render_template('index.html', search_an_image=search_an_image, s_filenames=s_filenames)


    filename = session.get('filename')
    if (filename != None):

        #print type(filename)
        #filename is string
        # transform it and copy it into the net
        print (os.getcwd())
        path_to_file = './uploads/'+filename
        image = caffe.io.load_image(path_to_file)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)

        # perform classification
        net.forward()

        # obtain the output probabilities
        output_prob = net.blobs['prob'].data[0]

        # sort top five predictions from softmax output
        top_inds = output_prob.argsort()[::-1][:5]


        #print 'probabilities and labels:'
        p_labels = []
        for i in range (0,5):
            tokens = labels[top_inds][i].split(' ')
            p_label = ''
            for j in range(1, len(tokens)):
                p_label = p_label + str(tokens[j])
            p_labels.append(p_label)

        p_and_tags = zip(output_prob[top_inds], p_labels)
        #print (zip(output_prob[top_inds], p_labels))
        #print (zip(output_prob[top_inds], labels[top_inds])[0])

        tag_data[filename] = p_labels
        print (tag_data)

        return render_template('index.html', filename = filename, p_and_tags=p_and_tags)
    
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        session['filename'] = filename
        print (filename)
    return redirect(url_for('index'))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload

#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#    print (filename)
#    return send_from_directory(app.config['UPLOAD_FOLDER'],
#                               filename)

if __name__ == "__main__":
    import click


    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='0.0.0.0')
    @click.argument('PORT', default=8111, type=int)
    def run(debug, threaded, host, port):

        HOST, PORT = host, port
        print "running on %s:%d" % (HOST, PORT)
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    
    run()
