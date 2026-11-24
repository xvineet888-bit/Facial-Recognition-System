const $ = id => document.getElementById(id);
const startBtn = $('startBtn'), stopBtn = $('stopBtn'), feed = $('videoFeed'), placeholder = $('placeholder');
async function start() {
  startBtn.disabled = true;
  $('recognitionText').textContent = 'Connecting to webcam…';
  try {
    const r = await fetch('/api/start', {method:'POST'});
    const data = await r.json();
    if (!data.ok) throw new Error(data.message || 'Could not start camera');
    feed.src = '/video?ts=' + Date.now();
    feed.style.display = 'block'; placeholder.style.display = 'none';
    stopBtn.disabled = false;
    $('systemState').textContent = 'Online';
    $('systemHint').textContent = 'Camera is running';
    $('liveTag').classList.add('on');
    $('liveTag').innerHTML = '<i></i> LIVE';
  } catch(e) {
    $('recognitionText').textContent = e.message;
    $('systemState').textContent = 'Offline';
  } finally { startBtn.disabled = false; }
}
async function stop() {
  stopBtn.disabled = true;
  await fetch('/api/stop', {method:'POST'});
  feed.removeAttribute('src'); feed.style.display = 'none'; placeholder.style.display = 'flex';
  $('systemState').textContent = 'Offline'; $('systemHint').textContent = 'Camera is stopped';
  $('liveTag').classList.remove('on'); $('liveTag').innerHTML = '<i></i> OFFLINE';
  $('recognitionText').textContent = 'Camera stopped';
  $('latestName').textContent = 'Waiting'; $('resultDescription').textContent = 'Start recognition to see results.';
  $('faceCount').textContent = '0'; $('resultFaces').textContent = '0';
}
startBtn.addEventListener('click', start); stopBtn.addEventListener('click', stop);
async function refresh() {
  try {
    const r = await fetch('/api/status'); const s = await r.json();
    if (s.running) {
      $('latestName').textContent = s.name || 'No face detected';
      $('recognitionText').textContent = s.name || 'No face detected';
      $('resultDescription').textContent = s.faces ? 'Identity result from the current camera frame.' : 'No face is currently visible in the frame.';
      $('faceCount').textContent = s.faces; $('resultFaces').textContent = s.faces;
      $('resultTime').textContent = s.updated || '—'; $('updatedAt').textContent = 'Last frame: ' + (s.updated || '—');
    }
  } catch(e) {}
}
setInterval(refresh, 700);
fetch('/api/identity-count').then(r=>r.json()).then(d=>$('identityCount').textContent=d.count).catch(()=>{$('identityCount').textContent='—'});
