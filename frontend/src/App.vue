<template>
  <div id="app" data="data" class="mt-0 mb-0">
    <b-container fluid style="width: 100%; min-height: 100vh" class="py-3 px-5">
      <b-row>
        <b-col xl="6" md="12" cols="12" class="flex-column d-flex">
          <b-row class="mb-1">
            <b-col>
              <TitleCard title="Angriffskarte"/>
            </b-col>
          </b-row>
          <b-row class="flex-grow-1" style="min-height: 400px">
            <b-col>
              <ShipMap :ships="$store.state.ships" :uboot_pos="$store.state.uboot_pos" :intercept="$store.state.intercept"/>
            </b-col>
          </b-row>
        </b-col>
        <b-col xl="3" md="6" cols="12">
          <b-row class="mb-1">
            <b-col>
              <TitleCard title="Ziele"/>
            </b-col>
          </b-row>
          <b-row>
            <b-col>
              <ShipList/>
            </b-col>
          </b-row>
        </b-col>
        <b-col xl="3" md="6" cols="12">
          <b-row class="mb-1">
            <b-col><TitleCard title="Einstellungen"/></b-col>
          </b-row>
          <b-row>
            <b-col>
              <TimeControl class="mb-2"/>
              <UBootInput class="mb-2"/>
              <InterceptControl
                  :ships="$store.state.ships"
                  :uboot_pos="$store.state.uboot_pos"
                  :time="this.$store.state.time"
                  :intercept="$store.state.intercept"
                  class="mb-2"
              />
            </b-col>
          </b-row>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import ShipList from './components/ShipLIst.vue'
import ShipMap from "./components/ShipMap";
import UBootInput from "@/components/UBootInput";
import TimeControl from "@/components/TimeControl";
import InterceptControl from "@/components/InterceptControl";
import TitleCard from "@/components/TitleCard";

export default {
  name: 'App',
  components: {
    TimeControl,
    ShipList,
    ShipMap,
    UBootInput,
    InterceptControl,
    TitleCard
  },
  created() {
    document.title = "U-Boot Rechner"
  }

}
</script>

<style>

#app{
  /*min-height: 100%;*/
  /*height: 100%;*/
  /*height: 100vh;*/
  background-image: url("../public/metal2.png");
}

body {
  margin: 0;
  /*height: 100%;*/
}

html {
  margin: 0;
  /*height: 100%;*/
}

.cardbackground {
  background-image: url("../public/brushed_alu.png")
}

/*#app {*/
/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/
/*  text-align: center;*/
/*  color: #2c3e50;*/
/*  margin-top: 60px;*/
/*}*/
</style>
