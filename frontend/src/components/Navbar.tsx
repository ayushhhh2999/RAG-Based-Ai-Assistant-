import { Sparkles } from "lucide-react";

const Navbar = () => {
  return (
    <nav className="glass border-b sticky top-0 z-50 backdrop-blur-lg">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="gradient-primary p-2 rounded-xl shadow-lg">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-bold gradient-text">
            Personal AI Assistant
          </h1>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
