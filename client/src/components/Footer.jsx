import { Github, Mail, FileText } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t border-border bg-card/50 backdrop-blur-md">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-orbitron font-bold mb-4 gradient-text">ProPredict</h3>
            <p className="text-sm text-muted-foreground">
              AI-powered protein function prediction using state-of-the-art computational biology models.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-secondary hover:bg-primary hover-glow transition-all"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="#"
                className="p-2 rounded-lg bg-secondary hover:bg-primary hover-glow transition-all"
              >
                <FileText className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
          <p>&copy; {new Date().getFullYear()} ProPredict. All rights reserved. MIT License.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
