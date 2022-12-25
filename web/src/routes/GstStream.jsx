import { h } from 'preact';
import JSMpegPlayer from '../components/JSMpegPlayer';
import Heading from '../components/Heading';

export default function GstStream() {
  return (
    <div className="space-y-4">
      <Heading size="2xl">floor0</Heading>
      <div>
        <JSMpegPlayer camera="floor0" />
      </div>
    </div>
  );
}
