from bottle import response

@app.route('/score')
def get_image():
    image = camera.capture()
    response.set_header('Content-type', 'image/jpeg')
    return image