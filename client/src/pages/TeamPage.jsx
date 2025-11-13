import React from "react";

const TeamPage = () => {
  const teamMembers = [
    {
      name: "Antas Dubey",
      role: "Project Lead & ML Engineer",
      image: "/team/Antas.png",
    },
    {
      name: "Archit Dubey",
      role: "ML Engineer",
      image: "/team/Archit.png",
    },
    {
      name: "Shivam",
      role: "Backend Developer",
      image: "/team/Shivam.png",
    },
    {
      name: "Vanshika Agarwal",
      role: "Frontend Developer",
      image: "/team/Vanshika.jpg",
    },
    {
      name: "Tanya Mishra",
      role: "Research Assistant",
      image: "/team/Tanya.png",
    },
  ];

  return (
    <main className="flex-1 pt-24 pb-12">
      <div className="animate-fadeIn max-w-4xl mx-auto px-4">

        {/* HEADER */}
        <h1 className="text-4xl md:text-5xl font-orbitron font-bold text-center mb-4">
          <span className="text-white">Meet the</span>{" "}
          <span className="gradient-text">Team</span>
        </h1>

        <p className="text-muted-foreground text-center mb-12">
          The scientists and engineers behind this project.
        </p>

        {/* TEAM MEMBERS — vertical & centered */}
        <div className="flex flex-col items-center gap-8 mb-16">
          {teamMembers.map((member, idx) => (
            <div
              key={idx}
              className="glass-card w-full max-w-xl p-6 rounded-lg hover-glow border border-border/40 bg-card/50 transition-all text-center"
            >
              <div className="flex flex-col items-center">
                
                {/* IMAGE TAG — replace with actual images later */}
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-24 h-24 rounded-full mb-4 border border-border object-cover"
                />

                <h3 className="text-xl font-bold text-foreground mb-1">
                  {member.name}
                </h3>

                <p className="text-primary font-semibold mb-2">
                  {member.role}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* TECHNOLOGIES SECTION */}
        <div className="glass-card p-8 rounded-lg hover-glow border border-border/40 bg-card/50">
          <h2 className="text-2xl font-orbitron font-bold mb-6 flex items-center gradient-text">
            <span className="text-3xl mr-3">🔧</span>
            Technologies & Tools
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { name: "ESM-2", desc: "Sequence Embeddings" },
              { name: "AlphaFold 2", desc: "Structure Prediction" },
              { name: "PyTorch", desc: "Deep Learning" },
              { name: "Scikit-learn", desc: "Ensemble Methods" },
              { name: "BioPython", desc: "Sequence Analysis" },
              { name: "React", desc: "Frontend Framework" },
              { name: "Node.js", desc: "Backend Runtime" },
              { name: "Tailwind CSS", desc: "UI Styling" },
            ].map((tech, idx) => (
              <div
                key={idx}
                className="glass-card p-4 rounded-lg bg-card hover:bg-accent/10 border border-border/30 transition-colors text-center"
              >
                <p className="font-semibold text-foreground">{tech.name}</p>
                <p className="text-xs text-muted-foreground mt-1">{tech.desc}</p>
              </div>
            ))}
          </div>
        </div>

      </div>
    </main>
  );
};

export default TeamPage;
