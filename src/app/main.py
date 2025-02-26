import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.protos', 'datagenerator', 'planetary_computer'))

import logging
import os
import sys
import time
import rasterio
import spacefx
import sys
from app_config import AppConfig
from image_processor import ImageProcessor
from spacefx.protos.common.Common_pb2 import StatusCodes
from spacefx.protos.sensor.Sensor_pb2 import SensorData
from PlanetaryComputer_pb2 import EarthImageRequest, EarthImageResponse, GeographicCoordinates

logger: logging.Logger = spacefx.logger(level=logging.INFO)

def on_receive_sensor_data(sensor_data):
    logger.info(f'Received sensor data:')
    logger.info(f'TrackingId: {sensor_data.responseHeader.trackingId}')
    logger.info(f'SensorID: {sensor_data.sensorID}')

    if sensor_data.sensorID != 'PlanetaryComputer':
        logger.info(f'SensorID: {sensor_data.sensorID} is not PlanetaryComputer. Ignoring.')
        return

    logger.info(f'Parsing PlanetaryComputer sensor data')
    sensor_payload = EarthImageResponse()
    sensor_payload.ParseFromString(sensor_data.data.value)
    logger.info('EarthImageResponse Sensor Data: %s', sensor_payload)

    geotiff_img = str(spacefx.link.get_xfer_directories()['inbox'])
    geotiff_img = f'{geotiff_img}/{sensor_payload.imageFiles[0].fileName}'

    logger.info(f'TrackingID: {sensor_data.responseHeader.trackingId}')
    logger.info(f'Asset: {sensor_payload.imageFiles[0].asset}')
    logger.info(f'Filename: {sensor_payload.imageFiles[0].fileName}')

    # Wait for the file to appear, using the associated linkResponse to indicate that the file is ready to use
    geotiff_img_linkresponse = f'{geotiff_img}.linkResponse'

    logger.info(f'Waiting for {geotiff_img_linkresponse} with 5 minutes timeout')
    file_available_timeout = time.time() + 300
    while (not os.path.isfile(geotiff_img_linkresponse)) and (time.time() < file_available_timeout):
        time.sleep(1)

    if not os.path.isfile(geotiff_img):
        logger.error(f'Failed to receive {geotiff_img_linkresponse}')
        raise TimeoutError(f'Failed to receive {geotiff_img}')

    logger.info(f'PlanetaryComputer Geotiff Image Received: {geotiff_img}.  Processing.')
    ImageProcessor.add_image_to_queue(geotiff_img)

def main():
    '''
    Initialize SpaceFx, , and submit a request to sensor
    '''
    logger.info('Initialize SpaceFX client')
    spacefx.client.build()

    app_config = AppConfig()

    for key, value in vars(app_config).items():
        logger.info(f'AppConfig {key} : {value}')

    logger.info('Subscribing to SensorData')
    spacefx.sensor.subscribe_to_sensor_data(callback_function=on_receive_sensor_data)

    logger.info('Querying sensors')
    sensors_avaliable = spacefx.sensor.get_available_sensors()
    for sensor in sensors_avaliable.sensors:
        logger.info('Sensors Found: %s', sensor.sensorID)

    if 'PlanetaryComputer' not in [sensor.sensorID for sensor in sensors_avaliable.sensors]:
        logger.info('PlanetaryComputer not found. Exiting.')
        sys.exit(1)

    logger.info(f'Tasking PlanetaryComputer sensor for ({app_config.LATITUDE}, {app_config.LONGITUDE})...')

    earth_image_request = EarthImageRequest()
    line_of_sight = GeographicCoordinates()
    line_of_sight.latitude = app_config.LATITUDE
    line_of_sight.longitude = app_config.LONGITUDE
    earth_image_request.collection = 'naip'
    earth_image_request.asset.append('image')
    earth_image_request.geographicCoordinates.CopyFrom(line_of_sight)

    payload_metadata = {'SOURCE_PAYLOAD_APP_ID': spacefx.client.get_app_id()}

    sensor_response = spacefx.sensor.sensor_tasking('PlanetaryComputer', earth_image_request, metadata=payload_metadata)
    
    logger.info('Sensor Tasking Request Status: %s', sensor_response.responseHeader.status)
    if sensor_response.responseHeader.status in [StatusCodes.PENDING, StatusCodes.SUCCESSFUL]:
        logger.info('Waiting for sensor data')
        spacefx.client.keep_app_open()
    else:
        logger.info('Exiting Application: Sensor PlanetaryComputer tasking was not successful')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
