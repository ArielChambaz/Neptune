export default function VideoPlayer() {
    return (
      <div className="relative pb-[56.25%] h-0 overflow-hidden rounded-lg shadow-lg">
        <img
          className="absolute top-0 left-0 w-full h-full object-cover rounded-lg"
          src="http://localhost:5000/api/video"
          alt="Processed feed"
        />
      </div>
    );
  }
  