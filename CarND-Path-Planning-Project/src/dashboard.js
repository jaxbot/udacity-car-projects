var ws;

function connect() {
	ws = new WebSocket("ws://localhost:4567");
	ws.onmessage = handleNewMessage;
	ws.onclose = reconnect;

	setInterval(requestNewData, 10);
}

function reconnect() {
	setTimeout(function() {
		connect();
	}, 1000);
}

function handleNewMessage(event) {
	var data = JSON.parse(event.data.substring(2))[1];
	console.log(data);

	document.getElementById("speed").innerHTML = data.speed;
	document.getElementById("lane").innerHTML = data.lane;
	document.getElementById("lane_speed").innerHTML = data.lane_speed;
	document.getElementById("state").innerHTML = data.state;
}

function requestNewData() {
	ws.send("42[\"dashboard\", {}]");
}

connect();
