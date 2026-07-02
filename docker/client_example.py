#!/usr/bin/env python3
"""
Minimal client for the Roma Medical inference service.

Sends one radiograph to a running service and prints the returned landmarks and
measurements. Requires only the ``requests`` package on the calling machine
(the heavy dependencies live inside the container, not here).

Examples:
    python client_example.py --image target.jpg --anatomy knee --projection lateral
    python client_example.py --image target.jpg --anatomy knee --projection lateral --mpp 0.148
    python client_example.py --image target.jpg --anatomy feet --projection ap --url http://gpu-server:8000
"""

import sys
import json
import argparse

import requests
import urllib3  # the service uses a self-signed cert; skip TLS verification on the internal net
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def process(url, image_path, anatomy, projection, mpp=None, api_key=None, timeout=600):
    # Optional: check the service is up and the model is loaded
    try:
        health = requests.get(f'{url}/health', timeout=10, verify=False).json()
        if not health.get('model_loaded'):
            print('Warning: service is up but the model is not loaded yet.', file=sys.stderr)
        print(f"Service: device={health.get('device')} cuda={health.get('cuda_available')}")
    except requests.RequestException as e:
        print(f'Could not reach {url}/health: {e}', file=sys.stderr)
        sys.exit(1)

    data = {'anatomy': anatomy, 'projection': projection}
    if mpp is not None:
        data['mpp'] = mpp

    headers = {'X-API-Key': api_key} if api_key else {}
    with open(image_path, 'rb') as f:
        resp = requests.post(
            f'{url}/process',
            files={'image': (image_path, f)},
            data=data,
            headers=headers,
            timeout=timeout,
            verify=False,
        )

    if not resp.ok:
        # The service returns a JSON {"detail": ...} on error
        try:
            detail = resp.json().get('detail', resp.text)
        except ValueError:
            detail = resp.text
        print(f'Request failed [{resp.status_code}]: {detail}', file=sys.stderr)
        sys.exit(2)

    return resp.json()


def main():
    parser = argparse.ArgumentParser(description='Call the Roma Medical inference service.')
    parser.add_argument('--image', required=True, help='Path to the target radiograph (jpg/png)')
    parser.add_argument('--anatomy', required=True, help='e.g. knee, feet, shoulder, hip')
    parser.add_argument('--projection', required=True, help='e.g. lateral, ap, axial')
    parser.add_argument('--mpp', type=float, default=None,
                        help='Millimeters per pixel (optional; uses the template default if omitted)')
    parser.add_argument('--url', default='http://localhost:8000',
                        help='Service base URL (e.g. https://your-server for the HTTPS stack)')
    parser.add_argument('--api-key', default=None, help='API key, if the service requires one')
    parser.add_argument('--out', default=None, help='Optional path to write the full JSON response')
    args = parser.parse_args()

    result = process(args.url, args.image, args.anatomy, args.projection, args.mpp, api_key=args.api_key)

    matching = result.get('matching', {})
    print(f"\nResult for {args.anatomy}/{args.projection} "
          f"(config '{result.get('config_tag')}', mpp={result.get('mpp')}):")
    print(f"  references used : {matching.get('num_references_used')}")
    print(f"  mean confidence : {matching.get('mean_confidence')}")
    print(f"  landmarks       : {len(result.get('landmarks', []))} points")
    print('  measurements    :')
    for name, value in result.get('measurements', {}).items():
        print(f"      {name}: {value}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull response written to {args.out}")


if __name__ == '__main__':
    main()
