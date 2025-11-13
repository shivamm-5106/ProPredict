import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand / Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center animate-pulse-glow">
              <span className="text-xl font-orbitron font-bold text-background">P</span>
            </div>
            <span className="text-xl font-orbitron font-bold gradient-text">
              ProPredict
            </span>
          </Link>

          {/* RIGHT SIDE EMPTY (navigation removed) */}
          <div></div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
