function [child] = mutation(child, Pm, mode)
Gene_no = length(child.Gene);

for k = 1: Gene_no

    % Check if mutation occurs
    R = rand();
    if R < Pm

        % Type of mutation
        if mode == "bitflip"
            child.Gene(k) = ~ child.Gene(k); % for binary

        elseif mode == "real01"
            child.Gene(k) = rand(); % [0,1] for real-valued    
        end

    end
end
    

end