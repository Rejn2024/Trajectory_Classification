-- Documented Tacview Lua API: paused, frame-by-frame engagement rendering.
-- Loaded only by the dedicated capture instance through /AddOnsFolder.
local Tacview = require("Tacview195")
local config = dofile(Tacview.AddOns.Current.GetPath() .. "config.lua")
local pending, drawn, lastId = nil, 0, -1
local clipIndex, startTime, pilot, opponent = nil, nil, nil, nil
local ownTransform, enemyTransform, enemyEnd, cameraRange
local markerQuad, markerBlack, markerWhite
local presentedId = -1
local heartbeatTime = 0

local function writeFile(name, value)
    local file = assert(io.open(config.work .. "/" .. name, "wb"))
    file:write(value)
    file:close()
end

local function loadClip(index)
    Tacview.Context.Playback.Pause()
    Tacview.Telemetry.Clear()
    assert(Tacview.Telemetry.Load(config.files[index]), "Replay load failed")
    startTime = Tacview.Telemetry.GetDataTimeRange()
    -- The API returns the beginning and end as separate values.
    if type(startTime) == "table" then startTime = startTime[1] end
    Tacview.Context.SetAbsoluteTime(startTime)
    pilot = Tacview.Telemetry.GetObjectHandle(0xA01, startTime)
    opponent = Tacview.Telemetry.GetObjectHandle(0xB01, startTime)
    local ignored
    ignored, enemyEnd = Tacview.Telemetry.GetTransformTimeRange(opponent)
    assert(pilot and pilot ~= 0, "Trained aircraft A01 missing")
    Tacview.Context.SetSelectedObject(0, 0)
    Tacview.Context.SetSelectedObject(1, 0)
    Tacview.Settings.SetString("UI.View.Camera.Mode", "Free")
    Tacview.Settings.SetBoolean("UI.View.Camera.Dogfight.Enabled", false)
    Tacview.Settings.SetString("UI.View.Terrain.Mode", "Full3D")
    for _, setting in ipairs({"Grid", "HUD", "Overlay", "Objects.Height", "Objects.LocalVectors", "Objects.Labels", "Objects.Heading", "Objects.Course", "Objects.LiftVector", "Objects.PilotLineOfSight"}) do
        Tacview.Settings.SetBoolean("UI.View." .. setting .. ".Visible", false)
    end
    Tacview.UI.EnterReadMode(true)
    Tacview.UI.SetWindowPosition(config.window_x, config.window_y, config.width, config.height)
    clipIndex = index
end

local function update(dt, absoluteTime)
    if os.time() ~= heartbeatTime then
        heartbeatTime = os.time()
        writeFile("heartbeat.txt", string.format("update %d lastId %.0f drawn %d", heartbeatTime, lastId, drawn))
    end
    local file = io.open(config.work .. "/request.txt", "rb")
    if file then
        local text = file:read("*a")
        file:close()
        local id, index, seconds = text:match("^(%d+) (%d+) ([%d%.%-]+)$")
        id = tonumber(id)
        if id and id ~= lastId then
            index, seconds = tonumber(index), tonumber(seconds)
            if index ~= clipIndex then loadClip(index) end
            Tacview.Context.Playback.Pause()
            Tacview.Context.SetAbsoluteTime(startTime + seconds)
            lastId, pending, drawn = id, id, 0
        end
    end
    if pilot then
        local now = Tacview.Context.GetAbsoluteTime()
        ownTransform = Tacview.Telemetry.GetTransform(pilot, now)
        enemyTransform = Tacview.Telemetry.GetTransform(opponent, math.min(now, enemyEnd))
        local a, b = ownTransform, enemyTransform
        local longitude = (a.longitude + b.longitude) / 2
        local latitude = (a.latitude + b.latitude) / 2
        local altitude = (a.altitude + b.altitude) / 2
        local north = (a.latitude - b.latitude) * 6371000
        local east = (a.longitude - b.longitude) * math.cos(latitude) * 6371000
        local separation = math.sqrt(north*north + east*east + (a.altitude-b.altitude)^2)
        cameraRange = math.max(separation * 1.45, 16000)
        local pitch, yaw = math.rad(40), math.rad(35)
        local horizontal = cameraRange * math.cos(pitch)
        Tacview.Context.Camera.SetSphericalPosition(
            longitude - horizontal * math.sin(yaw) / (6371000 * math.cos(latitude)),
            latitude - horizontal * math.cos(yaw) / 6371000,
            altitude + cameraRange * math.sin(pitch))
        Tacview.Context.Camera.SetRotation(0, -pitch, yaw)
        Tacview.Context.Camera.SetFieldOfView(math.rad(50))
    end
    -- PowerSave prevents the Update callback idling, but does not itself dirty
    -- the 3D viewport. Explicitly request every settling draw while paused.
    if pending then Tacview.UI.Update() end
end

local function drawAircraft()
    if not ownTransform then return end
    -- Enlarged native aircraft models make both pilots legible at BVR distances;
    -- their positions and attitudes remain the recorded telemetry values.
    ownTransform.scale = cameraRange / 400
    Tacview.UI.Renderer.DrawOpaqueObject(pilot, ownTransform)
    if Tacview.Context.GetAbsoluteTime() <= enemyEnd then
        enemyTransform.scale = cameraRange / 400
        Tacview.UI.Renderer.DrawOpaqueObject(opponent, enemyTransform)
    end
end

local function onDraw()
    if pending then
        drawn = drawn + 1
        if drawn >= 1 then presentedId = lastId end
    end
    if not markerQuad then
        markerQuad = Tacview.UI.Renderer.CreateVertexArray({0,0,0, 1,0,0, 0,1,0, 1,1,0})
        markerBlack = Tacview.UI.Renderer.CreateRenderState({color=0xff000000})
        markerWhite = Tacview.UI.Renderer.CreateRenderState({color=0xffffffff})
    end
    if presentedId >= 0 then
        local code = presentedId % 16777216
        for bit = 0,23 do
            local state = (math.floor(code / 2^bit) % 2 == 1) and markerWhite or markerBlack
            Tacview.UI.Renderer.DrawUIVertexArray({x=10+bit*6,y=Tacview.UI.Renderer.GetHeight()-18,scaleX=6,scaleY=6}, state, markerQuad)
        end
    end
    if pending then
        -- This callback follows the completed 3D pass for the requested state.
        -- Waiting for more identical draws can stall Tacview's idle renderer.
        if drawn >= 1 then
            local ax, ay = Tacview.UI.Renderer.GetProjectedPosition(pilot)
            local bx, by = Tacview.UI.Renderer.GetProjectedPosition(opponent)
            writeFile("ready.txt", string.format("%d %.6f %d %d %.3f %.3f %.3f %.3f", pending,
                Tacview.Context.GetAbsoluteTime() - startTime,
                Tacview.UI.Renderer.GetWidth(), Tacview.UI.Renderer.GetHeight(), ax or -1, ay or -1, bx or -1, by or -1))
            pending = nil
        end
    end
end

local function guarded(callback)
    return function(...)
        local ok, err = pcall(callback, ...)
        if not ok then writeFile("error.txt", tostring(err)) end
    end
end

Tacview.AddOns.Current.SetTitle("Offline replay video capture")
Tacview.AddOns.Current.SetVersion("1.0")
Tacview.AddOns.Current.SetNotes("Dedicated capture instance; never changes the ACMI source files.")
Tacview.Events.Update.RegisterListener(guarded(update))
Tacview.Events.DrawOpaqueObjects.RegisterListener(guarded(drawAircraft))
Tacview.Events.DrawTransparentUI.RegisterListener(guarded(onDraw))
Tacview.Events.PowerSave.RegisterListener(function() return false end)
Tacview.UI.EnterReadMode(true)
Tacview.UI.SetWindowPosition(config.window_x, config.window_y, config.width, config.height)
writeFile("loaded.txt", "Capture add-on loaded")
