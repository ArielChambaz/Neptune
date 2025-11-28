#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neptune Stream Router
WebSocket streaming endpoints for real-time video processing
"""

import logging
import json
import asyncio
import cv2
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming with detection
    
    Client sends video frames as base64 encoded images
    Server responds with detection results
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        detection_service = websocket.app.state.detection_service
        
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'frame':
                # Decode base64 image
                img_data = base64.b64decode(message['data'])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid frame data'
                    })
                    continue
                
                # Detect persons
                detections, proc_time = detection_service.detect_persons_in_image(
                    frame,
                    conf_threshold=message.get('conf_threshold')
                )
                
                # Send results
                await websocket.send_json({
                    'type': 'detections',
                    'detections': [det.dict() for det in detections],
                    'processing_time_ms': proc_time,
                    'frame_id': message.get('frame_id', 0)
                })
            
            elif message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
            
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f"Unknown message type: {message.get('type')}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


import numpy as np  # Add this at the top of the file
