function [child1 , child2] = crossover(parent1 , parent2, Pc, crossoverName)
% Christoph Lohrmann, LUT University, Finland
% Modified from Version by Seyedali Mirjalili

switch crossoverName
    case 'single' % One-point crossover
        Gene_no = length(parent1.Gene);
        ub = Gene_no - 1;
        lb = 1;
        Cross_P = round (  (ub - lb) *rand() + lb  );
        
%         Part1 = parent1.Gene(1:Cross_P);
%         Part2 = parent2.Gene(Cross_P + 1 : Gene_no);
        child1.Gene = [parent1.Gene(1:Cross_P), parent2.Gene(Cross_P + 1 : Gene_no)];
        
%         Part1 = parent2.Gene(1:Cross_P);
%         Part2 = parent1.Gene(Cross_P + 1 : Gene_no);
        child2.Gene = [parent2.Gene(1:Cross_P), parent1.Gene(Cross_P + 1 : Gene_no)];
        
        
    case 'double' % Two-point crossover
        Gene_no = length(parent1);
       
        ub = length(parent1.Gene) - 1;
        lb = 1;
        Cross_P1 = round (  (ub - lb) *rand() + lb  );
        
        Cross_P2 = Cross_P1;
        
        while Cross_P2 == Cross_P1
            Cross_P2 = round (  (ub - lb) *rand() + lb  );
        end
        
        if Cross_P1 > Cross_P2
            temp =  Cross_P1;
            Cross_P1 =  Cross_P2;
            Cross_P2 = temp;
        end
        Part1 = parent1.Gene(1:Cross_P1);
        Part2 = parent2.Gene(Cross_P1 + 1 :Cross_P2);
        Part3 = parent1.Gene(Cross_P2+1:end);
        
        child1.Gene = [Part1 , Part2 , Part3];
        
        
        Part1 = parent2.Gene(1:Cross_P1);
        Part2 = parent1.Gene(Cross_P1 + 1 :Cross_P2);
        Part3 = parent2.Gene(Cross_P2+1:end);
        
        child2.Gene = [Part1 , Part2 , Part3];
        
    case 'uniform' % uniform crossover
        Gene_no = length(parent1);    
        child1 = parent1;
        COidx = rand(Gene_no,1)<Pc; % elements affected by crossover (indices)
        child1(COidx) = parent2(COidx);
        child2 =[]; % not relevant

end

% Determine for each child if crossover is applied (<= Pc) to generate it or parent is retained
if rand() <= Pc
    child1 = child1;
    child2 = child2;
else
    child1 = parent1;
    child2 = parent2;
end

end

