# Gradio Whisper Demo Using FastAPI

This directory contains files for lunching Whisper demo based on Gradio interface.
The Whisper model is being use by API.


The files rquired for running the demo are above.
To lunch the demo, you should run both <code>client_run.py</code> (lunch the Gradio interface) and <code>server_run.py</code> (lunch the inference app).

<br>

In <code>client_run.py</code>, parameters of the app and of the model are defined.
<code>host</code> and <code>port</code> are params of the app. To run the demo on RAMBO, <code>host</code> should be <code>'0.0.0.0</code>.
<code>size</code> and <code>device</code> are related to the Whisper model that being used for inference. The default values are <code>medium</code> and <code>cuda</code> respectively.


In <code>server_run.py</code>, the URLS for live and static inference are defined. In addition, <code>debug_flag</code> is set. This parameter related to the debug option of the Gradio interface.

For using with RAMBO, the relevant URLS are -
<ul>
	<li> Static - <code>http://10.53.140.33:80/gradio_demo_static/</code>
	<li> Live - <code>http://10.53.140.33:80/gradio_demo_live/</code>
</ul>

For using on the same device, the relevant URLS are - 
<ul>
	<li> Static - <code>http://127.0.0.1:5000:5000/gradio_demo_static/</code>
	<li> Live - <code>http://127.0.0.1:5000:5000/gradio_demo_live/</code>
</ul>
Where <code>5000</code> is the port that being used by the app.

<br>
<br>
Other files in this directory are - 

<ul>
  <li><code>demo_utils.py</code> - Contains all needed functions for processing the audio and creates the transcription and VAD plot.</li>
  <li><code>FastAPI_gradio.py</code> - The gradio demo is defined.</li>
  <li><code>FastAPI_server.py</code> - The API app is defined.</li>
  <li><code>settings.py</code> - Initilaize all the global variables.</li>
</ul>
<img style="text-align: center; max-width: 650px; margin: 0 auto;"
                src="https://geekflare.com/wp-content/uploads/2022/02/speechrecognitionapi.png", 
                width="1500" height="300">
