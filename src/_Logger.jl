__precompile__(false)

module _Logger

using Printf

export info, emph, debug, PRINT_TO_CONSOLE, LOG_FILE, DEBUG_ENABLED
export print_as_exponent, f2, f4, greene

PRINT_TO_CONSOLE = true
DEBUG_ENABLED = false
LOG_FILE = nothing

function info(message::String, log_path=nothing)
    PRINT_TO_CONSOLE && println(message)

    append_to_log(LOG_FILE, message)
    append_to_log(log_path, message)
end


function emph(message::String, log_path=nothing)
    green_message = "\e[94m" * message * "\e[0m" 

    if PRINT_TO_CONSOLE
        println(green_message)
    end

    append_to_log(LOG_FILE, message)
    append_to_log(log_path, message)
end

function debug(message::String)
    if DEBUG_ENABLED
        PRINT_TO_CONSOLE && println(message)
        append_to_log(LOG_FILE, message)
    end
end

function append_to_log(file_path::Union{Nothing, String}, message::String)
    if file_path !== nothing
        open(file_path, "a") do file
            write(file, message * "\n")
        end
    end
end

function print_as_exponent(x::Int)::String
    if x == 0
        return "0"
    elseif log10(x) % 1 == 0  # Sprawdzamy, czy x jest dokładną potęgą 10
        exponent = floor(Int, log10(x))
        return @sprintf("10^%d", exponent)
    else
        return string(x)  # Jeśli nie jest potęgą 10, zwracamy zwykłą reprezentację
    end
end

function f4(x::Float64)::String
    return @sprintf("%.4f", x)
end

function f2(x::Float64)::String
    return @sprintf("%.2f", x)
end

function greene(text::String)::String
    return "\e[92m" * text * "\e[0m"
end

end
