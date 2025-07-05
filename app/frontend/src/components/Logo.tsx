import logo from "../assets/neptune_logo.png";

export default function Logo() {
  return (
    <div className="w-[400px] max-w-full p-4 flex items-center justify-center gap-4">
      <img src={logo} alt="Logo Neptune" className="block w-full" />
      <span className="text-6xl font-bold text-white">Neptune</span>
    </div>
  );
}
