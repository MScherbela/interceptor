<template>
  <div id="app" data="data">
    <b-container>
      <b-row>
        <b-col cols="8">

        </b-col>
        <b-col>
        <ShipList/>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import ShipList from './components/ShipLIst.vue'

export default {
  name: 'App',
  components: {
    ShipList
    // HelloWorld
  },
}
</script>

<style>
/*#app {*/
/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/
/*  text-align: center;*/
/*  color: #2c3e50;*/
/*  margin-top: 60px;*/
/*}*/
</style>
