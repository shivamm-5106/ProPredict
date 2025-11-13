import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, Dna, Zap, Shield } from "lucide-react";
import ProteinHelix from "@/components/ProteinHelix";
import { Suspense } from "react";


const Home = () => {
  return (
    <div className="min-h-screen flex flex-col">
      {/* <Navbar /> */}

      {/* Hero Section */}
      <section className="relative pt-24 pb-12 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-card to-background opacity-50" />
        
        <div className="container mx-auto px-4 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-5xl md:text-6xl font-orbitron font-bold leading-tight">
                Predict the <span className="gradient-text">Unseen</span>
              </h1>
              <p className="text-xl text-muted-foreground">
                Understand protein function from amino acid sequences using cutting-edge AI models like ESM-2 and AlphaFold 2.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/predict">
                  <Button 
                    size="lg" 
                    className="bg-primary hover:bg-primary-glow hover-glow text-background font-semibold group"
                  >
                    Predict
                    <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </Link>
                <Link to="/analysis">
                  <Button 
                    size="lg" 
                    variant="outline"
                    className="border-primary/50 hover:bg-secondary"
                  >
                    Analyse
                  </Button>
                </Link>
              </div>
            </div>

            <div className="h-[400px] lg:h-[500px] animate-fade-in">
                <Suspense fallback={<div>Loading 3D...</div>}><ProteinHelix /></Suspense>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-card/30">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-orbitron font-bold text-center mb-12 gradient-text">
            Why ProPredict?
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="glass-card p-6 rounded-lg hover-glow transition-all animate-fade-in">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
                <Dna className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Advanced AI Models</h3>
              <p className="text-muted-foreground">
                Powered by ESM-2 and AlphaFold 2, delivering state-of-the-art accuracy in protein function prediction.
              </p>
            </div>

            <div className="glass-card p-6 rounded-lg hover-glow transition-all animate-fade-in" style={{ animationDelay: '0.1s' }}>
              <div className="w-12 h-12 rounded-lg bg-accent/20 flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-accent" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Lightning Fast</h3>
              <p className="text-muted-foreground">
                Get comprehensive Gene Ontology predictions in seconds, not hours. Optimized for speed and efficiency.
              </p>
            </div>

            <div className="glass-card p-6 rounded-lg hover-glow transition-all animate-fade-in" style={{ animationDelay: '0.2s' }}>
              <div className="w-12 h-12 rounded-lg bg-secondary/20 flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Research Grade</h3>
              <p className="text-muted-foreground">
                Built on peer-reviewed methods and validated against extensive protein databases for reliable results.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10" />
        <div className="container mx-auto px-4 relative z-10 text-center">
          <h2 className="text-3xl md:text-4xl font-orbitron font-bold mb-6">
            Ready to Explore Protein Functions?
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Start predicting protein functions from sequences today. No installation required.
          </p>
          <Link to="/predict">
            <Button 
              size="lg" 
              className="bg-primary hover:bg-primary-glow hover-glow text-background font-semibold"
            >
              Start Predicting Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* <Footer /> */}
    </div>
  );
};

export default Home;
