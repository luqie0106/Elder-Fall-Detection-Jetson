navigator.mediaDevices.enumerateDevices()
  .then(devices => {
    devices.forEach(device => {
      if (device.kind === 'videoinput') {
        console.log(`设备名: ${device.label}, ID: ${device.deviceId}`);
      }
    });
  });